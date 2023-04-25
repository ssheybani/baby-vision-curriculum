#!/usr/bin/env python
# coding: utf-8

# Building off of exp_v2.1_seq.py
# - predictive loss


import sys, os, inspect

if 'BigRed200' in os.getcwd().split('/'):
#     print('Running on BigRed200')
    sys.path.insert(0,'/N/slate/hhansar/hfenv/lib/python3.10/site-packages')
SCRIPT_DIR = os.path.realpath(os.path.dirname(inspect.getfile(inspect.currentframe()))) #os.getcwd() #
# print('cwd: ',SCRIPT_DIR)
#os.path.realpath(os.path.dirname(inspect.getfile(inspect.currentframe())))
util_path = os.path.normpath(os.path.join(SCRIPT_DIR, '..', 'util'))
sys.path.insert(0, util_path)
os.environ['OPENBLAS_NUM_THREADS'] = '50' #@@@@ to help with the num_workers issue
os.environ['OMP_NUM_THREADS'] = '50'    
import numpy as np
import torch, torchvision
# import torchvision.transforms as tr
from torchvision import transforms as tr
from torch import nn
from torch.nn import functional as F
import os
import random
#import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from pathlib import Path
import math
import argparse
import pandas as pd
import warnings

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from ddputils import is_main_process, save_on_master, setup_for_distributed
import transformers

from PIL import Image
from torch.utils.data import Dataset

def get_fpathlist(vid_root, subjdir, ds_rate=1):
    """
    # read the image files inside vid_root/subj_dir into a list. 
    # makes sure they're all jpg. also sorts them so that the order of the frames is correct.
    # subjdir = ['008MS']
    """
    
    fpathlist = sorted(list(Path(os.path.join(vid_root, subjdir)).iterdir()), 
                       key=lambda x: x.name)
    fpathlist = [str(fpath) for fpath in fpathlist if fpath.suffix=='.jpg']
    fpathlist = fpathlist[::ds_rate]
    return fpathlist


class ImageSequenceDataset(Dataset):
    """
    To use for video models. 
    """
    def __init__(self, image_paths, transform):
        # transform is a Hugging Face image processor transform. check the usage in __getitem
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load the sequence of images
        images = [Image.open(fp) for fp in self.image_paths[idx]]
        images = self.transform(images, return_tensors="pt").pixel_values[0]
        return images
    
def get_train_val_split(fpathlist, val_ratio=0.1):
    """
    Splits the list of filepaths into a train list and test list
    """
    n_fr = len(fpathlist)
    val_size = int(n_fr*val_ratio)
    
    split1_idx = int((n_fr-val_size)/2)
    split2_idx = int((n_fr+val_size)/2)
    train_set =fpathlist[:split1_idx]+fpathlist[split2_idx:]
    val_set = fpathlist[split1_idx:split2_idx]
    return train_set, val_set

def get_fpathseqlist(fpathlist, seq_len, ds_rate=1, n_samples=None):
    """
    Returns a list of list that can be passed to ImageSequenceDataset
    # n_samples: int
    # between 1 and len(fpathlist)
    # If None, it's set to len(fpathlist)/seq_len
    """
    
    sample_len = seq_len*ds_rate
    if n_samples is None:
        n_samples = int(len(fpathlist)/seq_len)
        sample_stride = sample_len
    else:
        assert type(n_samples)==int
        sample_stride = int(len(fpathlist)/n_samples)

    fpathseqlist = [fpathlist[i:i+sample_len:ds_rate] 
                    for i in range(0, n_samples*sample_stride, sample_stride)]
    return fpathseqlist



def _get_transform(image_size):
# image_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
    image_processor = transformers.VideoMAEImageProcessor(
        size={"shortest_edge":image_size},
        do_center_crop=True, crop_size={"height":image_size, "width": image_size}
    )
    return image_processor



def make_dataset(subj_dirs, **kwargs):
    seq_len = kwargs['seq_len']
    n_groupframes=kwargs['n_groupframes']#1450000
    ds_rate = kwargs['ds_rate']
    jpg_root = kwargs['jpg_root']
    image_size = kwargs['image_size']
    transform = _get_transform(image_size)
    gx_fpathlist = []
    for i_subj, subjdir in enumerate(tqdm(subj_dirs)):
        gx_fpathlist += get_fpathlist(jpg_root, subjdir, ds_rate=ds_rate)
    gx_fpathlist = gx_fpathlist[:n_groupframes]

    # Train-val split
    gx_train_fp, gx_val_fp = get_train_val_split(gx_fpathlist, val_ratio=0.1)


    gx_train_fpathseqlist = get_fpathseqlist(gx_train_fp, seq_len, ds_rate=1, n_samples=None)
    gx_val_fpathseqlist = get_fpathseqlist(gx_val_fp, seq_len, ds_rate=1, n_samples=None)
    
    return {'train':ImageSequenceDataset(gx_train_fpathseqlist, transform=transform),
           'val': ImageSequenceDataset(gx_val_fpathseqlist, transform=transform)}

# Instantiate the model, and the optimizer
#----------------------------------
# def _adapt_model_simclr(model, n_features, n_out): 
#     model.fc = torch.nn.Sequential(
#         torch.nn.Linear(n_features, n_out), 
#         torch.nn.ReLU(), 
#         torch.nn.Linear(n_out, n_out)) 
#     _ = model.float()
#     return model



# def get_model(backbone_arg):
#     if backbone_arg =='res50':
#         xmodel = torchvision.models.resnet50(pretrained=False)
#         n_features, n_out = 2048, 2048
#     elif backbone_arg =='res18':
#         xmodel = torchvision.models.resnet18(pretrained=False)
#         n_features, n_out = 512, 2048
#     xmodel = _adapt_model_simclr(xmodel, n_features, n_out)
#     return xmodel

def get_model(image_size, num_frames, hidden_size, intermediate_size, num_attention_heads):
    config = transformers.VideoMAEConfig(image_size=image_size, patch_size=16, num_channels=3,
                                         num_frames=num_frames, tubelet_size=2, 
                                         hidden_size=hidden_size, num_hidden_layers=12, num_attention_heads=num_attention_heads,
                                         intermediate_size=intermediate_size, initializer_range=0.02,
                                         use_mean_pooling=True, decoder_num_attention_heads=6,
                                         decoder_hidden_size=384, decoder_num_hidden_layers=4, 
                                         decoder_intermediate_size=1536, norm_pix_loss=True)
    # config
    model = transformers.VideoMAEForPreTraining(config)
    return model
# def get_criterion(temperature=0.1):

    # return partial(info_nce_loss, temperature)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    



def setup(rank, world_size):    
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'    
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()
    

def DDP_process(rank, world_size, args, verbose=True):#protocol, seed):
    """
    

    Parameters
    ----------
    rank : TYPE
        GPU ID
    world_size : TYPE
        n GPU.
    protocol : TYPE
        as used by make_subjnames()
    seed : int
        0 1 or 2. Used for : Distributed Sampler, and protocol
        

    Returns
    -------
    None.

    """
    protocol = args.prot
    seed = args.data_seed
    n_epoch = args.n_epoch
    script_arg = args.script


    torch.cuda.set_device(rank)
    
    # directory names etc
    #---------------
    if len(args.savedir)==0:
        raise ValueError
        # model_dir = r"/N/scratch/sheybani/trainedmodels/multistage/simclr/aug23/"
    else:
        model_dir = args.savedir
    
    setup(rank, world_size) # setup the process groups
    print('Workers assigned.')
    is_main_proc = is_main_process()
    print('is_main_proc', is_main_proc) #@@@@
    setup_for_distributed(is_main_proc) #removes printing for child processes.
    
    
    
    other_seed = args.other_seed #np.random.randint(1000)
    torch.manual_seed(other_seed)
    
    
    number_of_cpu = len(os.sched_getaffinity(0))#multiprocessing.cpu_count() #joblib.cpu_count()
    

    
    # Instantiate the model, optimizer
    # xmodel = get_model('res50')
    image_size = 224
    num_frames = 16
    hidden_size = 768 #384
    intermediate_size = 3072 #4*384
    num_attention_heads = 12 #6
    xmodel = get_model(image_size=image_size, num_frames=num_frames, hidden_size=hidden_size, num_attention_heads=num_attention_heads, intermediate_size=intermediate_size)
    num_patches_per_frame = (xmodel.config.image_size // xmodel.config.patch_size) ** 2
    model_seq_length = (num_frames // xmodel.config.tubelet_size) * num_patches_per_frame
    mask_ratio = 0.9
    num_masks = int(mask_ratio * model_seq_length)
    xmodel = xmodel.to(rank)
    # print("model device", xmodel.device)
    xmodel = DDP(xmodel, device_ids=[rank], output_device=rank, 
                   find_unused_parameters=False)
    
    lr=1e-3 #1e-2#5e-3#1e-4
    wd =5e-5 # 1e-4

#     optimizer = torch.optim.Adam(xmodel.parameters(),
#                                 lr=lr,
#                                 weight_decay=wd)
    optimizer = torch.optim.Adam(xmodel.parameters(), lr=lr, weight_decay=wd)

    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                        # T_max=max_steps,
                                                        # eta_min=lr/100)
    
    # Instantiate the criterion, transform
    # criterion = get_criterion()
    # transform_homeview = get_transform(aug_arg)
    # transform_contrastive_hv = ContrastiveTransformations(transform_homeview, 
    #                                                    n_views=2)
    sampler_shuffle = True #@@@@ True #for the distributed dampler
    # Set the other hyperparams: n_epoch, batch_size, num_workers
    num_epochs = n_epoch #per train stage
    batch_size = 16 #128 #For individual GPUs
    pin_memory = True
    num_workers = 7#int((number_of_cpu-1)/4) #2 #0#1#2#3 #
    
    print('n cpu: ', number_of_cpu, ' n workers: ', num_workers)
    
    print('Model, optimizer, etc instantiated')
    
    print('seed: ',seed)
    # Split the data
    
    # if protocol is rand
    # if protocol=='rand':
        # subjnames = make_subjnames_random()
    
    # rand_select=False
    # subjnames = make_subjnames(protocol, seed,
                               # shape=groups_shape, rand_select=rand_select, 
                               # vid_root=None, subject_groups=None)
    
    # chunk_size = 1*60*5 #15*60*5 #15mins. will remain fixed. # reduced to 1min for the sequential.
    
    # n_chunks = 20*15 #20 #5h worth of video, 
    # n_frames = n_chunks*chunk_size #@@@ sequential: use all frames
    # sample_strategy='ds' #'rand'
    # fname_parts = get_fnames(subjnames,
    #                          n_frames=150000, shuffle=False, vid_root=None)
    # fname_parts = get_fnames_v2_1(subjnames, n_chunks, n_frames, 
                                  # chunk_size, sample_strategy=sample_strategy, vid_root=None)
    
    # print('data is split. subj_parts =', subjnames)
    # print('n_frames = ', len(fname_parts[0]), 
                             # len(fname_parts[1]), 
                                 # len(fname_parts[2])
                                # )
    
    #@@@@ related to sequential sampling
    # n_views = 2 #note that for n_views>2, the info_nce_loss() must be updated.
    jpg_root='/N/project/infant_image_statistics/preproc_saber/JPG_10fps/'
    ds_rate = 1

    n_groupframes = 1450000 # minimum number of frames across age groups

    g0='008MS+009SS_withrotation+010BF_withrotation+011EA_withrotation+012TT_withrotation+013LS+014SN+015JM+016TF+017EW_withrotation'
    g1='026AR+027SS+028CK+028MR+029TT+030FD+031HW+032SR+033SE+034JC_withlighting'
    g2='043MP+044ET+046TE+047MS+048KG+049JC+050AB+050AK_rotation+051DW'
# Total number of frames in each age group: g0=1.68m, g1=1.77m, g2=1.45m

    g0 = g0.split('+')
    g1 = g1.split('+')
    g2 = g2.split('+')                                               
    # for i_p, fname_part in enumerate(fname_parts):
        # perform the training
        # ideally could be replaced with a line like this:
            # model, results = train_ddp(model, fnames, optimizer, criterion, 
            #     n_ep, world_size, rank)
        
        # make the dataset, sampler and dataloader
        # dataset = SimpleImageDataset(fname_part, transform_contrastive_hv)
    # dataset = SequentialHomeviewDataset(fname_part, n_views, t_neigh, 
                                            # transform=transform_homeview)
    seq_len = num_frames #equivalent to num_frames in VideoMAE()
    #     ds_rate = 1
    n_samples = None#10 #50000
    datasets = make_dataset(g2, seq_len=seq_len, jpg_root=jpg_root, ds_rate=ds_rate, n_groupframes=n_groupframes, image_size=image_size)
    # sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, 
                                     # shuffle=sampler_shuffle, seed=seed)
    # batch_size = 1
    samplers_dict = {x: DistributedSampler(datasets[x], num_replicas=world_size, 
                                           rank=rank, shuffle=sampler_shuffle, 
                                           seed=seed)
                     for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(
        datasets[x], batch_size=batch_size, pin_memory=pin_memory, 
        num_workers=num_workers, shuffle=False, sampler=samplers_dict[x],
        prefetch_factor=int(1.5*batch_size/num_workers), drop_last=True)
                        for x in ['train', 'val']}
    # dataloader = torch.utils.data.DataLoader(
    #     dataset, batch_size=batch_size, pin_memory=pin_memory, 
    #     num_workers=num_workers, shuffle=False, sampler=sampler,
    #     prefetch_factor=int(1.5*batch_size/num_workers))


    print('len dset, len dloader: ', len(datasets['train']), len(dataloaders['train']))
#         print(dataset.__getitem__(22).shape)
    print('dataloaders created') #@@@
        
#         i_iter = 0 #ignore the one from the dataloader since we're counting beyond one epoch.
    verbose = (verbose and is_main_process())
    
    # data = {
    #     'results': {'train_acc':[],
    #                 'val_acc':[]}
    # }

    # outputs = [None for _ in range(world_size)]
    
    train_loss_history = []
    val_loss_history = []
    
    # best_model_wts = deepcopy(model.state_dict())
    best_loss = float('inf')
    
    
    for i_ep in range(num_epochs):
        if verbose:
            print('Epoch {}/{}'.format(i_ep, num_epochs - 1))
            print('-' * 10)
        # Each epoch has a training and validation phase
        
        for phase in ['train', 'val']:
            dataloaders[phase].sampler.set_epoch(i_ep)
            if phase == 'train':
                xmodel.train()  # Set model to training mode
            else:
                xmodel.eval()   # Set model to evaluate mode
            running_loss = torch.tensor([0.0], device=rank)

            for inputs in tqdm(dataloaders[phase]):
                print(inputs.shape)
#         break
                optimizer.zero_grad()
                num_zeros = model_seq_length - num_masks
                num_ones = num_masks

                bool_masked = np.zeros((batch_size, num_zeros + num_ones))
                bool_masked[:, :num_ones] = 1

                # Shuffle each array in the batch
                for i in range(bool_masked.shape[0]):
                    np.random.shuffle(bool_masked[i])
                bool_masked_pos = torch.from_numpy(bool_masked).bool()
                bool_masked_pos = bool_masked_pos.to(rank)
                # print("bool_masked_pos device", bool_masked_pos.get_device())
                inputs = inputs.to(rank)
                # print("input device", inputs.get_device())
                # bool_masked_pos = torch.randint(0, 2, (batch_size, model_seq_length)).bool()
                outputs = xmodel(inputs, bool_masked_pos=bool_masked_pos)
                loss = outputs.loss
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
        # break
                running_loss += loss.item() * inputs.size(0)
                

            dist.barrier() #@@@ synchronize all ranks. to avoid thread race condition at the beginning of the next epoch or when saving the model.
            dist.all_reduce(running_loss, op=dist.ReduceOp.SUM, async_op=False)
            
    #store the model and results to disk

                
            if is_main_proc:
                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                if verbose:
                    print('{} Loss: {:.4f}'.format(phase, epoch_loss.item()))
                if phase == 'val' and epoch_loss < best_loss:
                    best_loss = epoch_loss
                        # best_model_wts = deepcopy(model.state_dict())
                if phase == 'val':
                    val_loss_history.append(epoch_loss.cpu().item())
                else:
                    train_loss_history.append(epoch_loss.cpu().item())        
            
            

    if verbose:
        print('Training complete')
        print('Best val loss: {:4f}'.format(best_loss.item()))

    results_df = pd.DataFrame({'epoch': np.arange(len(train_loss_history)),
                               'train_loss':train_loss_history,
                               'val_loss': val_loss_history})

    if is_main_process():
        results_fpath = os.path.join(model_dir,"train_val_scores_g2")
        results_df.to_csv(results_fpath, sep=',', float_format='%.4f')
        # the model
        STAGE = "g2"
        model_fname = '_'.join(['model', protocol, 'stage', STAGE,
                                'seed',str(seed), 'other', str(other_seed)])+'.pt'
        MODELPATH = os.path.join(model_dir,model_fname)
        SCRIPT = script_arg
        # TRAIN_SETS = str(subjnames)

        torch.save({
                'stage': STAGE,
                'model_state_dict': xmodel.module.state_dict(),
                'script': SCRIPT,
                # 'train_sets': TRAIN_SETS
                }, MODELPATH)
#         'optimizer_state_dict': optimizer.state_dict(),
        print('model saved at ',MODELPATH)

    cleanup()
    
    
if __name__ == '__main__':
    
    #---------- Parse the arguments
    parser = argparse.ArgumentParser(description='Train Network on HeadCam Data')

    # Add the arguments
    parser.add_argument('--prot',
                           type=str,
                           help='The order of developmental stage to train the model on')

    parser.add_argument('--data_seed',
                           type=int,
                           help='')
    
    parser.add_argument('--savedir',
                           type=str,
                           help='directory to save the results')
    
    # parser.add_argument('-aug',
    #                        type=str,
    #                        help='apply data augmentation or not:y or n')
    
    parser.add_argument('--other_seed',
                           type=int,
                           help='A seed used as both model ID and a random seed')
    
    # parser.add_argument('--n_iter',
    #                        type=int,
    #                        default=0,
    #                        help='')
        
    parser.add_argument('--n_epoch',
                           type=int,
                           default=2,
                           help='')

    # parser.add_argument('--t_neigh',
    #                        type=int,
    #                        default=2,
    #                        help='neighborhood for selecting the positive sample')
    
    # parser.add_argument('--data_groups_shape',
    #                        type=str,
    #                        default='3,3',
    #                        help='')

    parser.add_argument('--script',
                           type=str,
                           default='',
                           help='An identifier for the script that generated the model,'+\
                        'e.g. smoothness_v1.2')
    

    #----------
    
# Execute the parse_args() method
    args = parser.parse_args()
    

    
    #--------------------------
    
    n_gpu = torch.cuda.device_count()
    world_size= n_gpu

    
    mp.spawn(
            DDP_process,
            args=(world_size, args),#prot_arg, seed_arg),
            nprocs=world_size
        )