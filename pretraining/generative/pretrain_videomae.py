import sys, os

# Place the root of the python environment
env_root = '/N/project/baby_vision_curriculum/pythonenvs/hfenv/lib/python3.10/site-packages/'
sys.path.insert(0, env_root)

os.environ['OPENBLAS_NUM_THREADS'] = '38' #@@@@ to help with the num_workers issue
os.environ['OMP_NUM_THREADS'] = '1' 


import numpy as np
import torch
# from torchvision import transforms as tr
from tqdm import tqdm
from pathlib import Path
# import math
import argparse
# import warnings

import transformers

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from ddputils import is_main_process, setup_for_distributed, AllReduce
from homeview import (_get_transform, get_fpathlist,
    get_fold, get_train_val_split, get_group, get_fpathseqlist, 
    ImageSequenceDataset, StillVideoDataset, make_dataset)
from mask import TubeMaskingGenerator, RandomMaskingGenerator
from controls import make_dataset_spatial
from loggingtools import (
        CSVLogger,
        grad_logger,
        AverageMeter)


# from PIL import Image
import random
import logging


def get_config(image_size, args):
    arch_kw = args.architecture
    
#     patch_size = 16
#     spatial_length = (image_size/patch_size)**2
#     temporal_length = args.num_frames/args.tubelet_size
#     sequence_length = spatial_length*temporal_length
    if arch_kw=='base': #default
        config = transformers.VideoMAEConfig(image_size=image_size, patch_size=16, num_channels=3,
                                             num_frames=args.num_frames, tubelet_size=args.tubelet_size, 
                                             hidden_size=768, num_hidden_layers=12, num_attention_heads=12,
                                             intermediate_size=3072, initializer_range=0.02,
                                             use_mean_pooling=True, decoder_num_attention_heads=6,
                                             decoder_hidden_size=384, decoder_num_hidden_layers=4, 
                                             decoder_intermediate_size=1536, norm_pix_loss=True)
    return config
    
    
def get_model(image_size, args):
    config = get_config(image_size, args)
    model = transformers.VideoMAEForPreTraining(config)
    return model

def init_model_from_checkpoint(model, checkpoint_path):
    # caution: model class
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def save_checkpoint(chpt_path, model, epoch, loss_meter,
                    batch_size, world_size, lr,
                    optimizer):
    save_dict = {
        'model_state_dict': model.module.state_dict(),
        'opt': optimizer.state_dict(),
        'epoch': epoch,
        'train_loss': loss_meter['train'].avg,
        'val_loss': loss_meter['val'].avg,
        'batch_size': batch_size,
        'world_size': world_size,
        'lr': lr
        }
    torch.save(save_dict, chpt_path)
    
def setup(rank, world_size):    
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()
    

def DDP_process(rank, world_size, args, verbose=True):
    
    train_group = args.train_group
    seed = args.seed
#     n_epoch = args.n_epoch
    # script_arg = args.script

    torch.cuda.set_device(rank)
    
    # directory names etc
    #---------------
    if len(args.savedir)==0:
        raise ValueError
    
    setup(rank, world_size) # setup the process groups
    print('Workers assigned.')
    
    # Access the memory capacity
    memory_capacity_bytes = torch.cuda.get_device_properties(rank).total_memory
    memory_capacity_gb = memory_capacity_bytes / (1024**3)

    if memory_capacity_gb<35:
        print('memory_capacity_gb:',memory_capacity_gb)
    
    is_main_proc = is_main_process()
    print('is_main_proc', is_main_proc)
    setup_for_distributed(is_main_proc) #removes printing for child processes.
    
    print('OPENBLAS_NUM_THREADS: ', os.environ['OPENBLAS_NUM_THREADS'])
    
    other_seed = args.seed 
    torch.manual_seed(other_seed)
    random.seed(args.seed)
    
    number_of_cpu = len(os.sched_getaffinity(0))#multiprocessing.cpu_count() #joblib.cpu_count()
    
    
    folder = args.savedir
    Path(folder).mkdir(parents=True, exist_ok=True)
    
    log_path = os.path.join(folder, 'csvlog_'+args.run_id+'.csv')
    
    chpt_fname = 'model_'+args.run_id+'.pth.tar'
    chpt_path = os.path.join(folder, chpt_fname)
    
        # -- make csv_logger
    csv_logger = CSVLogger(log_path,
                           ('%d', 'epoch'),
                           ('%d', 'itr'),
                           ('%.5f', 'train loss'),
                           ('%.5f', 'val loss'),
                           ('%.4e', 'grad-EFL'),
                           ('%.4e', 'grad-ELL'),
                           ('%.4e', 'grad-DLL'))
    logging.basicConfig()
    logger = logging.getLogger()
    if rank == 0:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)
        
    # Instantiate the model, optimizer
    image_size = 224
    use_bfloat16 = True 
    
    xmodel = get_model(image_size, args)
    
    if args.init_checkpoint_path!='na':
        print('args.init_checkpoint_path:',args.init_checkpoint_path)
        # initialize the model using the checkpoint
        xmodel = init_model_from_checkpoint(xmodel, args.init_checkpoint_path)
        
    # seq_len = xmodel.config.num_frames #equivalent to num_frames in VideoMAE()
    num_patches_per_frame = (xmodel.config.image_size // xmodel.config.patch_size) ** 2
    model_seq_length = (xmodel.config.num_frames // xmodel.config.tubelet_size) * num_patches_per_frame
    mask_ratio = args.mask_ratio
    # num_masks = int(mask_ratio * model_seq_length)
    
    nseq_time = xmodel.config.num_frames // xmodel.config.tubelet_size
    nseq_space = xmodel.config.image_size // xmodel.config.patch_size
    
    xmodel = xmodel.to(rank)
    # print("model device", xmodel.device)
    xmodel = DDP(xmodel, device_ids=[rank], output_device=rank, 
                   find_unused_parameters=False)
    
    lr=args.lr
    wd =args.wd
    momentum=args.momentum
    
    if args.optim=='sgd':
        optimizer = torch.optim.SGD(xmodel.parameters(), lr=lr, weight_decay=wd, 
                                    momentum=momentum, nesterov=True)
    elif args.optim=='adamw':
        optimizer = torch.optim.AdamW(xmodel.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.95))
    elif args.optim=='adam':
        optimizer = torch.optim.Adam(xmodel.parameters(), lr=lr, weight_decay=wd)
    else:
        raise ValueError('invalid argument for optim')
        
    scaler = torch.cuda.amp.GradScaler() if use_bfloat16 else None
    
    sampler_shuffle = True # True #for the distributed dampler
    # Set the other hyperparams: n_epoch, batch_size, num_workers
    num_epochs = args.n_epoch #per train stage
    batch_size = args.batch_size #16 #128 #For individual GPUs
    pin_memory = False
    num_workers = 6#2#int((number_of_cpu-1)/4) #2 #0#1#2#3 #
#     prefetch_factor = None #int(1.5*batch_size/num_workers)

    print('mask_ratio:', mask_ratio, 'lr:', lr) #@@@ Debug
    
    print('n cpu: ', number_of_cpu, ' n workers: ', num_workers)
    
    print('Model, optimizer, etc instantiated')
    
    print('seed: ',seed)
    group = get_group(train_group)
    print(group)
    if args.condition in ['MatchedSpatial', 'MatchedSpatioTemporal']:
        datasets = make_dataset_spatial(group, image_size, args)
    else:
        datasets = make_dataset(group, image_size, args)
    
    print('datasets[train][0].shape:',datasets['train'][0].shape) #@@@ debugging
    # sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, 
                                     # shuffle=sampler_shuffle, seed=seed)
    # batch_size = 1
    samplers_dict = {x: DistributedSampler(datasets[x], num_replicas=world_size, 
                                           rank=rank, shuffle=sampler_shuffle, 
                                           seed=seed)
                     for x in ['train', 'val']
                     if datasets[x] is not None}
    dataloaders = {x: torch.utils.data.DataLoader(
        datasets[x], batch_size=batch_size, pin_memory=pin_memory, 
        num_workers=num_workers, shuffle=False, sampler=samplers_dict[x],
        drop_last=True)
                        for x in ['train', 'val']
                        if datasets[x] is not None}

    
    
    maskgen_input_size = nseq_time, nseq_space, nseq_space
    mask_ratio = 0.9
    num_patches_per_frame
    if args.mask_sampler=='tube':
        mask_gen = TubeMaskingGenerator(maskgen_input_size, mask_ratio)
    else:
        mask_gen = RandomMaskingGenerator(maskgen_input_size, mask_ratio)
        
    print('len dset, len dloader: ', len(datasets['train']), len(dataloaders['train']))
#         print(dataset.__getitem__(22).shape)
    print('dataloaders created') #@@@
        
    verbose = (verbose and is_main_process())
    
#     STAGE = 1
    start_epoch = 0

    # --
    # log_timings = True
    log_freq = 10
    
    for epoch in range(start_epoch, start_epoch+num_epochs):
        if verbose:
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)
        # Each epoch has a training and validation phase
        
        loss_meter = {x:AverageMeter()
                      for x in ['train', 'val']}
        
        for phase in ['train', 'val']:
            if datasets[phase] is None:
                continue # for when we don't want a validation phase
            dataloaders[phase].sampler.set_epoch(epoch)
            if phase == 'train':
                xmodel.train()  # Set model to training mode
            else:
                xmodel.eval()   # Set model to evaluate mode
            
            i_iter=0 #,print_interval =0,10
#             monitor_interval = 200
            
            dloader_len = len(dataloaders[phase])
            if args.max_epoch_iters==0:
                n_epoch_iters = dloader_len
            else:
                n_epoch_iters = min(args.max_epoch_iters, dloader_len)
            print('n_epoch_iters: ',n_epoch_iters)
            
            
            for itr, inputs in enumerate(tqdm(dataloaders[phase])):
#                 print(inputs.shape)           
     
                def forward_loss(inputs):
                    # Shuffle each array in the batch
                    bool_masked = np.zeros((batch_size, model_seq_length))
                    for i_el in range(batch_size):
                        bool_masked[i_el,:] = mask_gen()
                    bool_masked_pos = torch.from_numpy(bool_masked).bool()
                    bool_masked_pos = bool_masked_pos.to(rank)
                    inputs = inputs.to(rank)
                    optimizer.zero_grad()
                    outputs = xmodel(inputs, bool_masked_pos=bool_masked_pos)
                    loss = outputs.loss
                    loss = AllReduce.apply(loss)
                    return loss

                with torch.cuda.amp.autocast(dtype=torch.bfloat16, 
                                             enabled=use_bfloat16):
                    loss = forward_loss(inputs)
                    
                if phase == 'train':
                    if use_bfloat16:
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        optimizer.step()
                    grad_stats = grad_logger(xmodel.module.named_parameters())
                    
                    
                loss_meter[phase].update(loss)
                
                def log_stats(phase):
                    if phase=='train':
                        train_loss = loss
                        val_loss = 0#np.nan
                    elif phase=='val':
                        train_loss = 0
                        val_loss = loss
                        grad_stats.enc_first_layer=0
                        grad_stats.enc_last_layer=0
                        grad_stats.dec_last_layer=0#np.nan
                    
                    csv_logger.log(epoch + 1, itr, 
                                   train_loss, val_loss, 
                                   grad_stats.enc_first_layer,
                                   grad_stats.enc_last_layer,
                                   grad_stats.dec_last_layer)
                    if (itr % log_freq == 0) or torch.isnan(loss) or torch.isinf(loss):
                        print('[%d, %5d] loss: %.3f '
                                    '[mem: %.2e] '
                                    % (epoch + 1, itr,
                                       loss_meter[phase].avg,
                                       torch.cuda.max_memory_allocated() / 1024.**2))

                        if grad_stats is not None:
                            logger.info('[%d, %5d] grad_stats: [%.2e %.2e %.2e]'
                                        % (epoch + 1, itr,
                                           grad_stats.enc_first_layer,
                                   grad_stats.enc_last_layer,
                                   grad_stats.dec_last_layer),)
                
                log_stats(phase)
                              
                i_iter+=1
                if i_iter>=n_epoch_iters:
                    break
                              
            if is_main_proc:
                logger.info('avg. loss %.3f' % loss_meter[phase].avg)
            
            dist.barrier() # synchronize all ranks. to avoid thread race condition at the beginning of the next epoch or when saving the model.
            
    if verbose:
        print('Training complete')

    if is_main_process():
        save_checkpoint(chpt_path,
                    xmodel,
                    epoch+1, loss_meter, 
                    batch_size, world_size, lr,
                    optimizer)
        
        print('All results saved at ', args.savedir)
        
        
    cleanup()
    
    
if __name__ == '__main__':
    
    #---------- Parse the arguments
    parser = argparse.ArgumentParser(description='Train Network on HeadCam Data')

    # Add the arguments
    parser.add_argument('-train_group',
                           type=str,
                           help='The age group on which the model gets trained. g0 or g1 or g2 or gr')

    parser.add_argument('-jpg_root',
                           type=str,
                           help='')
    
    parser.add_argument('-savedir',
                           type=str,
                           help='directory to save the results')
    
    parser.add_argument('-init_checkpoint_path',
                           type=str,
                           default='na',
                           help='')
    
    parser.add_argument('--mask_sampler',
                           type=str,
                           default='tube',
                           help='tube or random')
    
    parser.add_argument('--mask_ratio',
                           type=float,
                           default=0.9,
                           help='')
    parser.add_argument('--ds_rate',
                           type=int,
                           default=1,
                           help='temporal downsampling of the video frames')
    parser.add_argument('--fold',
                           type=int,
                           default=0,
                           help='for 30fps, which of the 3 folds of data to use')
    
    parser.add_argument('--optim',
                           type=str,
                           default='sgd',
                           help='')
    
    parser.add_argument('--lr',
                           type=float,
                           default=0.1,
                           help='')
    parser.add_argument('--wd',
                           type=float,
                           default=0,
                           help='')
    parser.add_argument('--momentum',
                           type=float,
                           default=0.9,
                           help='')
    parser.add_argument('--batch_size',
                           type=int,
                           default=16,
                           help='')
    parser.add_argument('--num_frames',
                           type=int,
                           default=16,
                           help='16 or 32')
    parser.add_argument('--tubelet_size',
                           type=int,
                           default=2,
                           help='temporal size of each patch')
    
    parser.add_argument('--architecture',
                           type=str,
                           default='',
                           help='see get_config')    
        
    parser.add_argument('--n_epoch',
                           type=int,
                           default=1,
                           help='')
    
    parser.add_argument('--n_trainsamples',
                           type=int,
                           default=81000,
                           help='how many train sample sequences to create (from the same number of frames). Controlled by the stride between the samples which can be shorter or longer than num_frames. 81000 would amount to stride=num_frames at 10fps with 1 data fold or 30fps and 3 data folds')
    
    parser.add_argument('--seed',
                           type=int,
                        default=0,
                           help='')

    parser.add_argument('--condition',
                           type=str,
                           default='default',
                           help='which control condition, e.g. static or shuffle or longshuffle')
    
    parser.add_argument('--max_epoch_iters',
                           type=int,
                           default=0,
                           help='0 is for unlimited, i.e. as many iters as the dataloader has')
    parser.add_argument('--run_id',
                           type=str,
                           default='',
                           help='An identifier for the saved files. suggested format ${curr}_${stage}_${train_group}_${condition}_${fold}_${seed}')
    
    parser.add_argument('--keep_val',
                           type=str,
                           default='n',
                           help='whether to set aside a validation set')

    parser.add_argument('--script',
                           type=str,
                           default='',
                           help='An identifier for the script that generated the model')
    

    #----------
    
# Execute the parse_args() method
    args = parser.parse_args()
    

    
    #--------------------------
    
    n_gpu = torch.cuda.device_count()
    world_size= n_gpu
    
    try:
        mp.spawn(
                DDP_process,
                args=(world_size, args),
                nprocs=world_size
            )
    except:
        cleanup()