#!/usr/bin/env python
# coding: utf-8

# Building off of exp_v2.1_seq.py
# - predictive loss


import sys, os, inspect

if 'BigRed200' in os.getcwd().split('/'):
#     print('Running on BigRed200')
    sys.path.insert(0,'/geode2/home/u080/sheybani/BigRed200/spenv/lib/python3.10/site-packages')
SCRIPT_DIR = os.path.realpath(os.path.dirname(inspect.getfile(inspect.currentframe()))) #os.getcwd() #
# print('cwd: ',SCRIPT_DIR)
#os.path.realpath(os.path.dirname(inspect.getfile(inspect.currentframe())))
util_path = os.path.normpath(os.path.join(SCRIPT_DIR, '..', 'util'))
sys.path.insert(0, util_path)    
import datautils

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
from copy import deepcopy
import argparse, pickle
from functools import partial
from itertools import chain

import warnings

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from ddputils import is_main_process, save_on_master, setup_for_distributed

from datautils import to_static, to_seq, to_batchseq, get_train_test_splits
from datautils import SequentialHomeviewDataset 

from make_homeview_dataset import make_subjnames, get_fnames_v2_1
# from make_homeview_dataset import SlownessTorchDatasetChunk, select_fnames
# from train_slowness import train_slowness_experimental, test_slowness


# Transform
#-----------------
class ContrastiveTransformations(object):

    def __init__(self, base_transforms, n_views=2):
        self.base_transforms = base_transforms
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transforms(x) for i in range(self.n_views)]
    
def get_transform(aug_arg):
    """
    Parameters
    ----------
    aug_arg : string
        may include any of the following:
        c: color augmentation
        b: blur augmentation
        o: orientation augmentation

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    transform_homeview : TYPE
        DESCRIPTION.

    """
    mean = [0.5, 0.5, 0.5]#np.mean(mean_all, axis=0) #mean_all[chosen_subj] 
    std = [0.25, 0.25, 0.25] #std_all[chosen_subj] 
    
    augs = [tr.Resize(224), tr.CenterCrop(224)]
    
    if 'c' in aug_arg:
        color_augs = [tr.RandomApply([tr.ColorJitter(0.9, 0.9, 0.9, 0.5)], p=0.9),
                      tr.RandomGrayscale(p=0.2)]
        augs += color_augs
    if 'b' in aug_arg:
        blur_augs = [tr.RandomApply(
            [tr.GaussianBlur(kernel_size=7, sigma=[0.1, 2.])],
            p=0.5)]
        augs += blur_augs
    if 'o' in aug_arg:
        orientation_augs = [tr.RandomHorizontalFlip(p=0.5),
                           tr.RandomRotation(degrees=(-90, 90))]
        augs += orientation_augs
        
    augs += [tr.ConvertImageDtype(torch.float32), 
             tr.Normalize(mean,std)]
    transform_homeview = tr.Compose(augs)
    return transform_homeview

# Instantiate the model, and the optimizer
#----------------------------------
def _adapt_model_simclr(model, n_features, n_out): 
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(n_features, n_out), 
        torch.nn.ReLU(), 
        torch.nn.Linear(n_out, n_out)) 
    _ = model.float()
    return model



def get_model(backbone_arg):
    if backbone_arg =='res50':
        xmodel = torchvision.models.resnet50(pretrained=False)
        n_features, n_out = 2048, 2048
    elif backbone_arg =='res18':
        xmodel = torchvision.models.resnet18(pretrained=False)
        n_features, n_out = 512, 2048
    xmodel = _adapt_model_simclr(xmodel, n_features, n_out)
    return xmodel


def get_criterion(temperature=0.1):

    return partial(info_nce_loss, temperature)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
def info_nce_loss(temperature, feats, mode='train'):
    # Calculate cosine similarity
    cos_sim = F.cosine_similarity(feats[:,None,:], feats[None,:,:], dim=-1)
    # Mask out cosine similarity to itself
    self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
#     cos_sim.masked_fill_(self_mask, -9e15)
    # Find positive example -> batch_size//2 away from the original example
    pos_mask = self_mask.roll(shifts=cos_sim.shape[0]//2, dims=0)
    
    neg_mask = torch.ones_like(pos_mask, dtype=torch.bool, device=cos_sim.device)
#     thresh = 0.8
#     neg_mask[cos_sim>thresh] = False
    neg_mask[pos_mask | self_mask] = False
#     print('sum(neg_mask): ', torch.sum(neg_mask).item())

    # InfoNCE loss
    cos_sim = cos_sim / temperature
    
    pos_part = -cos_sim[pos_mask]
    neg_part = torch.logsumexp(cos_sim[neg_mask], dim=-1) # + #cos_sim, dim=-1)
    # print('loss pos, loss_neg = ', pos_part.mean().item(), neg_part.mean().item())
    nll = pos_part + neg_part
    nll = nll.mean()

    return nll


def setup(rank, world_size):    
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'    
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()
    

def DDP_process(rank, world_size, args):#protocol, seed):
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
    aug_arg = args.aug
    seed = args.data_seed
    n_epoch = args.n_epoch
    groups_shape = tuple([int(item) 
                          for item in args.data_groups_shape.split(',')])
    t_neigh = args.t_neigh
    # model_id_arg = args.model_id
#         t_neigh = args.t_neigh
    script_arg = args.script


    torch.cuda.set_device(rank)
    data = {
        'results': {'train_loss':[]}   
    }
    # we have to create enough room to store the collected objects
#     outputs = [None for _ in range(world_size)]
    
    
    # directory names etc
    #---------------
    if len(args.savedir)==0:
        raise ValueError
        # model_dir = r"/N/scratch/sheybani/trainedmodels/multistage/simclr/aug23/"
    else:
        model_dir = args.savedir

    os.environ['OPENBLAS_NUM_THREADS'] = '10' #@@@@ to help with the num_workers issue
    
    
    setup(rank, world_size) # setup the process groups
    print('Workers assigned.')
    is_main_proc = is_main_process()
    print('is_main_proc', is_main_proc) #@@@@
    setup_for_distributed(is_main_proc) #removes printing for child processes.
    
    
    
    other_seed = args.other_seed #np.random.randint(1000)
    torch.manual_seed(other_seed)
    
    
    number_of_cpu = len(os.sched_getaffinity(0))#multiprocessing.cpu_count() #joblib.cpu_count()
    

    
    # Instantiate the model, optimizer
    xmodel = get_model('res50')
    xmodel = xmodel.to(rank)
    xmodel = DDP(xmodel, device_ids=[rank], output_device=rank, 
                   find_unused_parameters=False)
    
    lr=1e-2#5e-3#1e-4
    wd = 1e-4
    
    n_groupiter = n_epoch*800 #args.n_iter #1000 #400
    
    n_totaliter = 3*n_groupiter #1500
    step_interval = 10
#     max_epochs = 3*n_epoch #used only for the scheduler
    max_steps = int(n_totaliter/step_interval)
#     optimizer = torch.optim.Adam(xmodel.parameters(),
#                                 lr=lr,
#                                 weight_decay=wd)
    optimizer = torch.optim.SGD(xmodel.parameters(),
                                lr=lr,
                                momentum=0.9,
                                weight_decay=wd)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                        T_max=max_steps,
                                                        eta_min=lr/100)
    
    # Instantiate the criterion, transform
    criterion = get_criterion()
    transform_homeview = get_transform(aug_arg)
    # transform_contrastive_hv = ContrastiveTransformations(transform_homeview, 
    #                                                    n_views=2)
    sampler_shuffle = True #@@@@ True #for the distributed dampler
    # Set the other hyperparams: n_epoch, batch_size, num_workers
    num_epochs = n_epoch #per train stage
    batch_size = 32 #128 #For individual GPUs
    pin_memory = True
    num_workers = 7#int((number_of_cpu-1)/4) #2 #0#1#2#3 #
    
    print('n cpu: ', number_of_cpu, ' n workers: ', num_workers)
    
    print('Model, optimizer, etc instantiated')
    
    print('seed: ',seed)
    # Split the data
    
    # if protocol is rand
    # if protocol=='rand':
        # subjnames = make_subjnames_random()
    
    rand_select=False
    subjnames = make_subjnames(protocol, seed,
                               shape=groups_shape, rand_select=rand_select, 
                               vid_root=None, subject_groups=None)
    
    chunk_size = 1*60*5 #15*60*5 #15mins. will remain fixed. # reduced to 1min for the sequential.
    
    n_chunks = 20*15 #20 #5h worth of video, 
    n_frames = n_chunks*chunk_size #@@@ sequential: use all frames
    sample_strategy='ds' #'rand'
    # fname_parts = get_fnames(subjnames,
    #                          n_frames=150000, shuffle=False, vid_root=None)
    fname_parts = get_fnames_v2_1(subjnames, n_chunks, n_frames, 
                                  chunk_size, sample_strategy=sample_strategy, vid_root=None)
    
    print('data is split. subj_parts =', subjnames)
    print('n_frames = ', len(fname_parts[0]), 
                             len(fname_parts[1]), 
                                 len(fname_parts[2])
                                )
    
    #@@@@ related to sequential sampling
    n_views = 2 #note that for n_views>2, the info_nce_loss() must be updated.
                                                 
    for i_p, fname_part in enumerate(fname_parts):
        # perform the training
        # ideally could be replaced with a line like this:
            # model, results = train_ddp(model, fnames, optimizer, criterion, 
            #     n_ep, world_size, rank)
        
        # make the dataset, sampler and dataloader
        # dataset = SimpleImageDataset(fname_part, transform_contrastive_hv)
        dataset = SequentialHomeviewDataset(fname_part, n_views, t_neigh, 
                                            transform=transform_homeview)
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, 
                                     shuffle=sampler_shuffle, seed=seed)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, pin_memory=pin_memory, 
            num_workers=num_workers, shuffle=False, sampler=sampler,
            prefetch_factor=int(1.5*batch_size/num_workers))
        print('len dset, len dloader: ', len(dataset), len(dataloader))
#         print(dataset.__getitem__(22).shape)
        print('dataloader created') #@@@
        
#         i_iter = 0 #ignore the one from the dataloader since we're counting beyond one epoch.
        for i_ep in range(num_epochs):
            # set epoch
#             dist.barrier()
            dataloader.sampler.set_epoch(i_ep)
            
            xmodel.train()
            for i_t, xbatch in enumerate(tqdm(dataloader)):
                optimizer.zero_grad()#set_to_none=True)
                xbatch = torch.cat(xbatch, dim=0)
                # print('xbatch.shape:',xbatch.shape) #@@@@
                pred = xmodel(xbatch)
#                 print('pred.shape:',pred.shape) #@@@@
                loss = criterion(pred)
                loss.backward()
                optimizer.step()
                
                if is_main_proc and i_t%10==0:
                    # record i_t, loss. 
                    print('Part: {} Epoch: {} Iteration: {} Train Loss : {}  '.format(i_p, i_ep, i_t, loss))
                    data['results']['train_loss'].append([i_p, i_ep, i_t, loss.cpu().item()])
                
                if lr_scheduler is not None:
                    if i_t%step_interval==0:
                        lr_scheduler.step()
                        print('lr = ', get_lr(optimizer))
                
#                 i_iter+=1
#                 if i_iter>=n_groupiter: #@@@
#                     break

            dist.barrier() #@@@ synchronize all ranks. to avoid thread race condition at the beginning of the next epoch or when saving the model.
            
            # call the validation script if need be
            
                
            
        
    #store the model and results to disk
#     dist.all_gather_object(outputs, data) # Gathers "data" from the whole group and puts it into "outputs". 
    if is_main_proc:
        
        # the model
        STAGE = i_p
        model_fname = '_'.join(['model', protocol, 'stage', str(i_p),
                                'seed',str(seed), str(other_seed)])+'.pt'
        MODELPATH = os.path.join(model_dir,model_fname)
        SCRIPT = script_arg
        TRAIN_SETS = str(subjnames)

        torch.save({
                'stage': STAGE,
                'model_state_dict': xmodel.module.state_dict(),
                'script': SCRIPT,
                'train_sets': TRAIN_SETS
                }, MODELPATH)
#         'optimizer_state_dict': optimizer.state_dict(),
        print('model saved at ',MODELPATH)
        
        
#         allresults = []
#         for cdict in outputs:
#             allresults += list(chain(*cdict['results']['train_loss'])) 

            # print(len(cdict['results']['train_loss']), len(allresults))
        allresults = data['results']['train_loss']
    
        results_fname = '_'.join(['results', protocol, 
                                  'seed',str(seed), str(other_seed)])+'.pkl'
        results_fpath = os.path.join(model_dir,results_fname)
        pickle.dump(allresults, 
                    open(results_fpath, "wb" ) )
        print('results saved at ',results_fpath)
#     torch.save(allresults, results_fpath)
    


    cleanup()
    
    
if __name__ == '__main__':
    
    #---------- Parse the arguments
    parser = argparse.ArgumentParser(description='Train Network on HeadCam Data')

    # Add the arguments
    parser.add_argument('-prot',
                           type=str,
                           help='The order of developmental stage to train the model on')

    parser.add_argument('-data_seed',
                           type=int,
                           help='')
    
    parser.add_argument('-savedir',
                           type=str,
                           help='directory to save the results')
    
    parser.add_argument('-aug',
                           type=str,
                           help='apply data augmentation or not:y or n')
    
    parser.add_argument('-other_seed',
                           type=int,
                           help='A seed used as both model ID and a random seed')
    
    parser.add_argument('--n_iter',
                           type=int,
                           default=0,
                           help='')
        
    parser.add_argument('--n_epoch',
                           type=int,
                           default=2,
                           help='')

    parser.add_argument('--t_neigh',
                           type=int,
                           default=2,
                           help='neighborhood for selecting the positive sample')
    
    parser.add_argument('--data_groups_shape',
                           type=str,
                           default='3,3',
                           help='')

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
