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

import transformers

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from ddputils import is_main_process, save_on_master, setup_for_distributed

from dsdatasets import SSv2Dataset, ToyboxDataset, make_ucf101dataset, ucf_collate, _get_transform, transform_vid
# from transformers import VideoMAEConfig, VideoMAEModel
# from torch.utils.data import Dataset
# import av

# from time import time
# from copy import deepcopy
# import cv2
# from itertools import chain

# ------------
# Dataset and Dataloader


def my_collate(batch):
    batch = tuple(filter(lambda x: x[0] is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


#------------------------------------
# Get Model
def get_config(image_size, args, num_labels=2):
    arch_kw = args.architecture
    
    if arch_kw=='base': #default
        config = transformers.VideoMAEConfig(image_size=image_size, patch_size=16, num_channels=3,
                                             num_frames=args.num_frames, tubelet_size=args.tubelet_size, 
                                             hidden_size=768, num_hidden_layers=12, num_attention_heads=12,
                                             intermediate_size=3072, num_labels=num_labels)
    else:
        raise ValueError
    return config


def init_model_from_checkpoint(model, checkpoint_path):
    # caution: model class
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def adapt_videomae(source_model, target_model):
    # load the embeddings
    target_model.videomae.embeddings.load_state_dict(
        source_model.videomae.embeddings.state_dict())
#     load the encoder
    target_model.videomae.encoder.load_state_dict(
        source_model.videomae.encoder.state_dict())
    return target_model

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
#         for param in model.classifier.parameters():
#             param.requires_grad = True
            
def get_model(image_size, num_labels, feature_extracting, init_checkpoint_path, args):
    config_source = get_config(image_size, args)
    model_source = transformers.VideoMAEForPreTraining(config_source)
    
    if init_checkpoint_path!='na':
        print('init_checkpoint_path:',init_checkpoint_path)
        # initialize the model using the checkpoint
        model_source = init_model_from_checkpoint(model_source, init_checkpoint_path)
  
    config_target = get_config(image_size, args, num_labels=num_labels)
    model_target = transformers.VideoMAEForVideoClassification(config=config_target)
#     model_target = transformers.VideoMAEModel(config=config_target) #@@@ do not add the classifer head
    model_target = adapt_videomae(model_source, model_target)
#     if not torch.all(
#         model_target.embeddings.patch_embeddings.projection.weight==model_source.videomae.embeddings.patch_embeddings.projection.weight):
#         warnings.warn('Model not successfully initialized')
    if not torch.all(
        model_target.videomae.embeddings.patch_embeddings.projection.weight==model_source.videomae.embeddings.patch_embeddings.projection.weight):
        warnings.warn('Model not successfully initialized')
    
#     if feature_extracting: #@@@@@ redundant with torch.no_grad
#         set_parameter_requires_grad(model_target, feature_extracting)
    
    return model_target

#------------------------------------


def save_results(fnames, embeddings, phase, run_id, args):
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
    if phase=='test':
        savedir = os.path.join(savedir,'test/')
    Path(savedir).mkdir(parents=True, exist_ok=True)
#         <model vs scores>_<prot>_seed_<seed>_other_<other>_<other id>
    result_fname = '_'.join(['embeddings', run_id])+'.csv'
    results_fpath = os.path.join(savedir, result_fname)
    xdf.to_csv(results_fpath, sep=',', float_format='%.6f', index=False)
    print('embeddings saved at ',results_fpath)
    

def get_run_id(fp):
    fname = Path(fp).name
    return fname.replace('model_','').replace('.pth.tar', '')
    
    
def setup(rank, world_size):    
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'    
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()
    
def DDP_process(rank, world_size, args, verbose=True):#protocol, seed):

    seed = args.seed
    
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
    print('Workers assigned.')
    is_main_proc = is_main_process()
    print('is_main_proc', is_main_proc) #@@@@
    setup_for_distributed(is_main_proc)
    
    number_of_cpu = len(os.sched_getaffinity(0))#multiprocessing.cpu_count() #joblib.cpu_count()
    
    #Instantiate the dataset, criterion
    transform = transform_vid
    frame_rate=args.frame_rate
    sample_len=args.num_frames
    
    if args.dataset_split=='both':
        phases = ['train','test']
    else:
        phases = [args.dataset_split]

    datasets = {}
    
    for x in phases:
        train = (x=='train')
        if args.ds_task=='ssv2':
            datasets[x] = SSv2Dataset(args.vid_root, transform, 
                              frame_rate=frame_rate, sample_len=sample_len,
                                                train=train)
        elif args.ds_task=='toybox':
            datasets[x] = ToyboxDataset(args.vid_root, transform, 
                              frame_rate=frame_rate, sample_len=sample_len,
                                                train=train)
        elif args.ds_task=='ucf101':
                datasets[x] = make_ucf101dataset(sample_len, frame_rate, num_workers=args.num_workers,
                                                train=train)
        else:
            raise ValueError

    if args.ds_task=='ucf101':
        collate_fn = ucf_collate
    else:
        collate_fn = my_collate
#     feature_extract = True

    # Instantiate the model, optimizer
    #Load the model, adapt it to the downstream task
    image_size = 224
#     feature_extract = (args.finetune=='n')
    num_classes=0
    feature_extract=True
    
    chpt_dir = args.checkpoint_dir
    if chpt_dir=='notUsed':
        fpaths = [args.init_checkpoint_path]
    else:
        fpaths = [str(Path(chpt_dir, fname))
                 for fname in os.listdir(chpt_dir)
                 if Path(chpt_dir, fname).suffix=='.tar']
    
    for fp in tqdm(fpaths):
        init_checkpoint_path = fp
        run_id = get_run_id(fp)
        xmodel = get_model(image_size, num_classes, feature_extract, 
                           fp, args)
    
        xmodel = xmodel.to(rank)
        xmodel = DDP(xmodel, device_ids=[rank], output_device=rank, 
                       find_unused_parameters=False)


        # Make the dataloaders and samplers
        sampler_shuffle = False #for the distributed dampler
        batch_size = args.batch_size# 128
        pin_memory = False
        num_workers = args.num_workers #number_of_cpu-1#32

    #     collate_fn = my_collate#None
        print('n cpu: ', number_of_cpu, ' n workers: ', num_workers)
        
        
#         break #@@@
        
        for phase in phases:
            dataset = datasets[phase]
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
                    inputs = inputs.to(rank)
                    image_features = xmodel(pixel_values=inputs).logits
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

                save_results(allfnames, allembeddings, phase, run_id, args)

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
    
    parser.add_argument('--checkpoint_dir',
                           type=str,
                        default='notUsed',
                           help='the directory of all checkpoints. Set if you want to try all checkpoints. init_checkpoint_path, run_id will be ignored.')
    
    parser.add_argument('--dataset_split',
                           type=str,
                           default='both',
                           help='one of train, test, both')
    
    parser.add_argument('--frame_rate',
                           type=int,
                           default=6,
                           help='frame rate of the videos in the benchmark')
    
    parser.add_argument('--num_frames',
                           type=int,
                           default=16,
                           help='length of the input across the time dim')
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
                           type=int,
                        default=0,
                           help='A seed used as both model ID and a random seed')
    
    parser.add_argument('--run_id',
                           type=str,
                        default='x',
                           help='protocol name. used only for naming the output files')
    #----------


    # Execute the parse_args() method
    args = parser.parse_args()
    
    
   #-------------
    
    n_gpu = torch.cuda.device_count()
    world_size= n_gpu

    try:
        mp.spawn(
                DDP_process,
                args=(world_size, args),#prot_arg, seed_arg),
                nprocs=world_size
            )
    except:
        cleanup()