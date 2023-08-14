"""
Report downstream performance scores for a pretrained model.
"""
import sys, os

env_root = '/N/project/baby_vision_curriculum/pythonenvs/hfenv/lib/python3.10/site-packages/'
sys.path.insert(0, env_root)

os.environ['OPENBLAS_NUM_THREADS'] = '38' #@@@@ to help with the num_workers issue
os.environ['OMP_NUM_THREADS'] = '1'  #10

import numpy as np
import torch, torchvision
from torchvision import transforms as tr
from tqdm import tqdm
from pathlib import Path
# import math
import argparse
import pandas as pd
import warnings

# import transformers

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from ddputils import is_main_process, save_on_master, setup_for_distributed

# import cv2
# script_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scripts')
# print(script_dir)
# sys.path.insert(0, script_dir)
from dsdatasets import SSv2Dataset, ToyboxDataset, make_ucf101dataset, make_cifar10dataset, ucf_collate, _get_transform, transform_vid

# import vision_transformer as vit
# from helper import load_checkpoint
from tensors import trunc_normal_

# ------------
# Dataset and Dataloader
def my_collate(batch):
    batch = tuple(filter(lambda x: x[0] is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


#------------------------------------
# Get Model

def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.LayerNorm):
        torch.nn.init.constant_(m.bias, 0)
        torch.nn.init.constant_(m.weight, 1.0)

def _adapt_model_simclr(model, n_features, n_out): 
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(n_features, n_out), 
        torch.nn.ReLU(), 
        torch.nn.Linear(n_out, n_out)) 
    _ = model.float()
    return model

def get_untrained_model(
    model_name='resnet50', pred_emb_dim=1024):
    xmodel = getattr(torchvision.models, model_name)()
#     xmodel = torchvision.models.resnet50(pretrained=False)
    n_features, n_out = 1*pred_emb_dim, pred_emb_dim
    xmodel = _adapt_model_simclr(xmodel, n_features, n_out)
    # xmodel = xmodel.to(device)
    return xmodel

def init_model_from_checkpoint(model, checkpoint_path):
    # caution: model class
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def get_model(args):
    
    encoder = get_untrained_model(model_name=args.architecture, pred_emb_dim=args.pred_emb_dim)
    
    load_path = args.init_checkpoint_path
    if load_path!='na':
        print('load_path:',load_path)
        encoder = init_model_from_checkpoint(encoder, load_path)
    encoder.fc = torch.nn.Identity() #@@@
    return encoder

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
#         for param in model.classifier.parameters():
#             param.requires_grad = True


#------------------------------------


def save_results(fnames, embeddings, args):
    print('embeddings.shape:',embeddings.shape)
    print('len(fnames):',len(fnames))
#     print('type(fnames):',type(fnames))
    try:
        print('type(fnames[0]):',type(fnames[0]))
    except:
        pass
#     if type(fnames[0])==list:
#         print('len(fnames[0]):',len(fnames[0]))
    hdim = embeddings.shape[1]
    xdf = pd.DataFrame(embeddings, columns= ['dim'+str(i)
                                         for i in range(hdim)])
    xdf['fnames'] = fnames
    xdf = xdf[['fnames']+ list(xdf.columns[:-1])]

    xdf = xdf.sort_values('fnames')
    xdf = xdf.drop_duplicates(subset='fnames', ignore_index=True)

    savedir = args.savedir
    Path(savedir).mkdir(parents=True, exist_ok=True)
#         <model vs scores>_<prot>_seed_<seed>_other_<other>_<other id>
    result_fname = '_'.join(['embeddings', args.run_id])+'.csv'
    results_fpath = os.path.join(savedir, result_fname)
    xdf.to_csv(results_fpath, sep=',', float_format='%.6f', index=False)
    print('embeddings saved at ',results_fpath)
    
    
def setup(rank, world_size):    
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'    
#     dist.init_process_group("nccl", rank=rank, world_size=world_size)
    dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()
    
def DDP_process(rank, world_size, args, verbose=True):#protocol, seed):

    seed = args.seed
    
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
    torch.manual_seed(seed)
        
    # data = {
    #     'results': {'train_loss':[]}   
    # }
    # we have to create enough room to store the collected objects
    # outputs = [None for _ in range(world_size)]
    
     # directory names etc
    #---------------
    
    # setup the process groups
    setup(rank, world_size) 
    
    # device = torch.device("cuda", rank) if torch.cuda.is_available() else torch.device("cpu")

        
    print('Workers assigned.')
    is_main_proc = is_main_process()
    print('is_main_proc', is_main_proc) #@@@@
    setup_for_distributed(is_main_proc)
    
    number_of_cpu = len(os.sched_getaffinity(0))#multiprocessing.cpu_count() #joblib.cpu_count()
    
    #Instantiate the dataset, criterion
    transform = transform_vid#_get_transform(224)#
    frame_rate=args.frame_rate
    sample_len=args.num_frames
    
    if args.ds_task=='ssv2':
        dataset = SSv2Dataset(args.vid_root, transform, 
                          frame_rate=frame_rate, sample_len=sample_len)
    elif args.ds_task=='toybox':
        dataset = ToyboxDataset(args.vid_root, transform, 
                          frame_rate=frame_rate, sample_len=sample_len)
    elif args.ds_task=='ucf101':
        dataset = make_ucf101dataset(sample_len, frame_rate, num_workers=args.num_workers)
    elif args.ds_task=='cifar10':
        dataset = make_cifar10dataset(sample_len)
    else:
        raise ValueError

    if args.ds_task=='ucf101':
        print('ucf101 Line185')
        collate_fn = ucf_collate
    else:
        collate_fn = my_collate
    
    
    # Instantiate the model, optimizer
    #Load the model, adapt it to the downstream task
#     image_size = 224
#     feature_extract = (args.finetune=='n')
#     num_classes=0
#     feature_extract=True
    xmodel = get_model(args)
    
    xmodel = xmodel.to(rank)
#     xmodel = xmodel.to(device)
    xmodel = DDP(xmodel, device_ids=[rank], output_device=rank, 
                   find_unused_parameters=False)
    
    
    # Make the dataloaders and samplers
    sampler_shuffle = False #for the distributed dampler
    batch_size = args.batch_size# 128
    pin_memory = False
    num_workers = args.num_workers #number_of_cpu-1#32
    
    collate_fn = my_collate#None
    print('n cpu: ', number_of_cpu, ' n workers: ', num_workers)
    
    sampler = DistributedSampler(dataset, num_replicas=world_size, 
                                           rank=rank, shuffle=sampler_shuffle, 
                                           seed=seed)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, pin_memory=pin_memory, collate_fn=collate_fn,
        num_workers=num_workers, shuffle=False, sampler=sampler, drop_last=False)
    print('len dset, len dloader: ', len(dataset), len(dataloader))
#         print(dataset.__getitem__(22).shape)
    print('dataloaders created') #@@@
    
    verbose = (verbose and is_main_process())
    
    
    data = {
        'fnames':[],
        'embeddings': []   
    }
    outputs = [None for _ in range(world_size)]

    print_period=1#20
#     i_break = 5
    with torch.no_grad():
        for i_t, xbatch in enumerate(tqdm(dataloader)):
            inputs, fnames = xbatch
            if inputs is None:
                continue
#             print('len(fnames):',len(fnames)) #@@@@@debug
#             print('inputs.shape:',inputs.shape)
            inputs = inputs[:,-1,...]        
            inputs = inputs.to(rank) #@@@
#             inputs = inputs.to(device) #@@@
#             B,T,C,H,W = inputs.shape
            #.view(B*T, C, H, W)
            image_features = xmodel(inputs)#.mean(1)
#             image_features /= image_features.norm(dim=-1, keepdim=True)
            data['fnames'] += fnames
            data['embeddings'].append(image_features.detach().cpu().numpy())

            if (i_t%print_period)==0:
                memory_allocated = torch.cuda.memory_allocated() / 1024**2
                print(f'GPU memory allocated: {memory_allocated:.2f} MB')
#             if i_t==i_break:
#                 break #@@@
    print('len(data[fnames]):',len(data['fnames'])) #@@@@@debug
    dist.all_gather_object(outputs, data)
    
            
    if is_main_process():
        print('finished processing')
        allfnames, allembeddings = [],[]
        for cdict in outputs:
            allfnames += cdict['fnames']
#             allfnames += list(chain(*cdict['fnames'])) 
            print('Aggregating worker results:',len(cdict['fnames']),'/', len(allfnames))
            allembeddings +=cdict['embeddings']
            
        allembeddings = np.concatenate(allembeddings)
    
        save_results(allfnames, allembeddings, args)
        

    cleanup()
    
    
#____________________________________________
if __name__ == '__main__':
    
    #---------- Parse the arguments
    parser = argparse.ArgumentParser(description='Evaluate downstream performance for a pretrained model.')

    # Add the arguments

    parser.add_argument('-ds_task',
                           type=str,
                           help='one of ssv2, toybox, ucf101')
        
    parser.add_argument('-vid_root',
                           type=str,
                           help='absolute path to the ssv2 dataset')
    
    parser.add_argument('-init_checkpoint_path',
                           type=str,
                           help='absolute path to the checkpoint file of the pretrained model')

    parser.add_argument('-savedir',
                           type=str,
                           default='',
                           help='')
    
    parser.add_argument('--frame_rate',
                           type=int,
                           default=3,
                           help='frame rate of the videos in the benchmark')
    
    parser.add_argument('--num_frames',
                           type=int,
                           default=16,
                           help='length of the input across the time dim')
    
    parser.add_argument('--pred_emb_dim',
                           type=int,
                           default=2048,
                           help='size of the embedding layer')
    
    parser.add_argument('--tubelet_size',
                           type=int,
                           default=2,
                           help='temporal size of each patch')
    parser.add_argument('--batch_size',
                           type=int,
                           default=64,
                           help='')
    parser.add_argument('--num_workers',
                           type=int,
                        default=6,
                           help='number of cpu for each gpu')
    
    parser.add_argument('--architecture',
                           type=str,
                           default='',
                           help='see get_config')    

    parser.add_argument('--seed',
                           type=str,
                        default=0,
                           help='')
        
    parser.add_argument('--run_id',
                           type=str,
                        default='x',
                           help='protocol name. used only for naming the output files')

    
    #----------


    # Execute the parse_args() method
    args = parser.parse_args()
    
    
   #-------------
    
    n_gpu = torch.cuda.device_count()
    
    world_size = n_gpu if n_gpu > 0 else 1  # If no GPUs, set world size to 1

#     world_size= n_gpu

    try:
        mp.spawn(
                DDP_process,
                args=(world_size, args),#prot_arg, seed_arg),
                nprocs=world_size
            )
    except:
        cleanup()