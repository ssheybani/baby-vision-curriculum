import sys, os

env_root = '/N/project/baby_vision_curriculum/pythonenvs/hfenv/lib/python3.10/site-packages/'
sys.path.insert(0, env_root)

os.environ['OPENBLAS_NUM_THREADS'] = '38' #@@@@ to help with the num_workers issue
os.environ['OMP_NUM_THREADS'] = '1'  #10


import numpy as np
import torch, torchvision
from tqdm import tqdm
from pathlib import Path
import argparse


import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

import random
from copy import deepcopy

from ddputils import is_main_process, save_on_master, setup_for_distributed, AllReduce
from homeview import (_get_transform, get_fpathlist,
    get_fold, get_train_val_split, get_fpath2framelist, 
    TwoFrameDataset, get_group)

from loggingtools import (
        CSVLogger,
        grad_logger,
        AverageMeter,
        gpu_timer)
import yaml
import torch.nn.functional as F

import logging
from functools import partial

        
        
def make_dataset(subj_dirs, image_size, args):
    ds_rate = args.ds_rate #kwargs['ds_rate']
    jpg_root = args.jpg_root #kwargs['jpg_root']
#     image_size = kwargs['image_size']
    fold = args.fold #kwargs['fold']
    condition = args.condition #kwargs['condition']
    n_trainsamples = args.n_trainsamples
    augs = args.augs
    crop_scale = (0.7, 1.)
    
    transform = _get_transform(image_size, augs=augs, crop_size=image_size, crop_scale=crop_scale)
    gx_fpathlist = []
    for i_subj, subjdir in enumerate(tqdm(subj_dirs)):
        gx_fpathlist += get_fpathlist(jpg_root, subjdir, ds_rate=ds_rate)
    
    max_folds = 3
    gx_fpathlist = get_fold(gx_fpathlist, fold, max_folds, args)
    print('Num. frames in the fold:',len(gx_fpathlist))

    if condition=='shuffle':
        random.shuffle(gx_fpathlist)
    
    gx_fpath2framelist = get_fpath2framelist(gx_fpathlist, args.interval, n_samples=n_trainsamples)
    train_dataset = TwoFrameDataset(gx_fpath2framelist, transform=transform)
        
    return {'train':train_dataset,
           'val': None}

def _adapt_model_simclr(model, n_features, n_out): 
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(n_features, n_out), 
        torch.nn.ReLU(), 
        torch.nn.Linear(n_out, n_out)) 
    _ = model.float()
    return model

def get_model(device, model_name='resnet50', pred_emb_dim=1024):
    xmodel = getattr(torchvision.models, model_name)()
    n_features, n_out = 1*pred_emb_dim, pred_emb_dim
    xmodel = _adapt_model_simclr(xmodel, n_features, n_out)
    xmodel = xmodel.to(device)
    return xmodel

def get_special_matrix(n):
    x = [
    [1 if i == j + 1 or i == j - 1 else 0 
     for j in range(n)] 
    for i in range(n)]
    return np.asarray(x)

def save_checkpoint(chpt_path, model, epoch, loss_meter,
                    batch_size, world_size, lr,
                    optimizer):
    save_dict = {
        'model_state_dict': model.module.state_dict(),
        'opt': optimizer.state_dict(),
        'epoch': epoch,
        'train_loss': loss_meter.avg,
        'batch_size': batch_size,
        'world_size': world_size,
        'lr': lr
        }
    torch.save(save_dict, chpt_path)
    

def init_model_from_checkpoint(model, checkpoint_path):
    # caution: model class
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def info_nce_loss(temperature, masks, feats, mode='train'):
    # Calculate cosine similarity
    cos_sim = F.cosine_similarity(feats[:,None,:], feats[None,:,:], dim=-1)
    
    pos_mask, neg_mask = masks

    # InfoNCE loss
    cos_sim = cos_sim / temperature

    pos_part = -cos_sim[pos_mask]
    neg_part = torch.logsumexp(cos_sim[neg_mask], dim=-1) 
    nll = neg_part + pos_part 
    nll = nll.mean()

    return nll

def setup(rank, world_size):    
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()
    

def DDP_process(rank, world_size, args, verbose=True):
    """
    

    Parameters
    ----------
    rank : TYPE
        GPU ID
    world_size : TYPE
        n GPU.
        

    Returns
    -------
    None.

    """
    train_group = args.train_group
    seed = args.seed
    torch.cuda.set_device(rank)
    
    logging.basicConfig()
    logger = logging.getLogger()
    if rank == 0:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)
    
    
    setup(rank, world_size) # setup the process groups
    print('Workers assigned.')
    
    is_main_proc = is_main_process()
    print('is_main_proc', is_main_proc) 
    setup_for_distributed(is_main_proc) #removes printing for child processes.
    
    print('OPENBLAS_NUM_THREADS: ', os.environ['OPENBLAS_NUM_THREADS'])
    
    other_seed = args.seed
    torch.manual_seed(other_seed)
    random.seed(args.seed)
    
    number_of_cpu = len(os.sched_getaffinity(0))#multiprocessing.cpu_count() #joblib.cpu_count()
    
    image_size = 224
    use_bfloat16 = True
    
    verbose = (verbose and is_main_process())
    
    # -- LOGGING
    folder = args.savedir 
    Path(folder).mkdir(parents=True, exist_ok=True)
    params_file = 'params_'+args.run_id+'.yaml'
    dump = os.path.join(folder, params_file)
    with open(dump, 'w') as f:
        yaml.dump(args, f)
    
    # -- log/checkpointing paths
    log_path = os.path.join(folder, 'csvlog_'+args.run_id+'.csv')
    chpt_fname = 'model_'+args.run_id+'.pth.tar'
    chpt_path = os.path.join(folder, chpt_fname)
    
    # -- make csv_logger
    csv_logger = CSVLogger(log_path,
                           ('%d', 'epoch'),
                           ('%d', 'itr'),
                           ('%.5f', 'loss'),
                           ('%.4e', 'grad-FL'),
                           ('%.4e', 'grad-LL'))
    
    # -- init model
    
    
    device= rank#'cuda:0'#'cpu'
    model_name=args.architecture #'vit_base'#'vit_small'
#     pred_depth=6
    pred_emb_dim=args.pred_emb_dim 
    xmodel = get_model(
        device=device,
        pred_emb_dim=pred_emb_dim,
        model_name=model_name)
    
    if args.init_checkpoint_path!='na':
        print('args.init_checkpoint_path:',args.init_checkpoint_path)
        # initialize the model using the checkpoint
        xmodel = init_model_from_checkpoint(xmodel, args.init_checkpoint_path)
    
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

    sampler_shuffle = True # True for the distributed dampler
    # Set the other hyperparams: n_epoch, batch_size, num_workers
    num_epochs = args.n_epoch #per train stage
    batch_size = args.batch_size #16 #128 #For individual GPUs
    pin_memory = False#True
    num_workers = 6#2#int((number_of_cpu-1)/4) #2 #0#1#2#3 #

    print('n cpu: ', number_of_cpu, ' n workers: ', num_workers)
    
    print('Model, optimizer, etc instantiated')
    
    print('seed: ',seed)
    group = get_group(train_group)
    print(group)                                               
    datasets = make_dataset(group, image_size, args)
    
    print('datasets[train][0].shape:',datasets['train'][0].shape) #@@@ debugging
    
    samplers_dict = {x: DistributedSampler(datasets[x], num_replicas=world_size, 
                                           rank=rank, shuffle=sampler_shuffle, 
                                           seed=seed)
                     for x in ['train']}#, 'val']}

    dataloaders = {x: torch.utils.data.DataLoader(
        datasets[x], batch_size=batch_size, pin_memory=pin_memory, 
        num_workers=num_workers, shuffle=False, sampler=samplers_dict[x],
        drop_last=True)
                        for x in ['train']}#, 'val']}

    
    print('len dset, len dloader: ', len(datasets['train']), len(dataloaders['train']))
    print('dataloaders created')
        
    start_epoch = 0

    log_freq = 10
    
#     Create the criterion
    temperature = 0.1
    mask_size = batch_size*2
    self_mask = torch.eye(mask_size, dtype=torch.bool, device=rank, requires_grad=False)
    pos_mask = torch.tensor(get_special_matrix(mask_size),
                            dtype=torch.bool, device=rank, requires_grad=False)
    neg_mask = torch.ones_like(pos_mask, dtype=torch.bool, device=rank, requires_grad=False)
    neg_mask[pos_mask | self_mask] = False
    masks = (pos_mask, neg_mask)
    criterion = partial(info_nce_loss, temperature, masks)
    
    
    phase = 'train'
    for epoch in range(start_epoch, start_epoch+num_epochs):
        if verbose:
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)
        loss_meter = AverageMeter()
        phase = 'train'
        dataloaders[phase].sampler.set_epoch(epoch)
        if phase == 'train':
            xmodel.train()  # Set model to training mode
        else:
            xmodel.eval()   # Set model to evaluate mode
        i_iter=0 #,print_interval =0,10
        
        dloader_len = len(dataloaders[phase])
        if args.max_epoch_iters==0:
            n_epoch_iters = dloader_len
        else:
            n_epoch_iters = min(args.max_epoch_iters, dloader_len)
        print('n_epoch_iters: ',n_epoch_iters)
        
        
        
        for itr, inputs in enumerate(tqdm(dataloaders[phase])):

            def forward_loss(inputs):
                # Shuffle each array in the batch
                inputs = inputs.to(rank)
                B,T,C,H,W = inputs.shape
                inputs = inputs.view(B*T,C,H,W)
                optimizer.zero_grad()
                pred = xmodel(inputs)
                loss = criterion(pred)
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


            loss_meter.update(loss)
            
            def log_stats():
                csv_logger.log(epoch + 1, itr, loss, 
                               grad_stats.first_layer,
                               grad_stats.last_layer)
                if (itr % log_freq == 0) or torch.isnan(loss) or torch.isinf(loss):
                    print('[%d, %5d] loss: %.3f '
                                '[mem: %.2e] '
                                % (epoch + 1, itr,
                                   loss_meter.avg,
                                   torch.cuda.max_memory_allocated() / 1024.**2))

                    if grad_stats is not None:
                        logger.info('[%d, %5d] grad_stats: [%.2e %.2e]'
                                    % (epoch + 1, itr,
                                       grad_stats.first_layer,
                                       grad_stats.last_layer,))
            log_stats()                
            i_iter+=1
            if i_iter>=n_epoch_iters:
                break
                              
        if is_main_proc:
            logger.info('avg. loss %.3f' % loss_meter.avg)

        dist.barrier() #@@@ synchronize all ranks. to avoid thread race condition at the beginning of the next epoch or when saving the model.
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
    
    parser.add_argument('--init_checkpoint_path',
                           type=str,
                           default='',
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
                           default=0.7,
                           help='')
    parser.add_argument('--batch_size',
                           type=int,
                           default=16,
                           help='')
    parser.add_argument('--interval',
                           type=int,
                           default=30,
                           help='interval between anchor and positive in the unit of steps (depends on ds_rate)')    
    parser.add_argument('--augs',
                           type=str,
                           default='n',
                           help='c: RandomResizedCrop, b:GaussianBlur, j:ColorJitter, g: RandomGrayscale')  
    parser.add_argument('--architecture',
                           type=str,
                           default='resnet50',
                           help='see get_config')    
    parser.add_argument('--pred_emb_dim',
                           type=int,
                           default=2048,
                           help='size of the embedding layer')
    
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
# loss_record_period
    parser.add_argument('--run_id',
                           type=str,
                           default='',
                           help='An identifier for the saved files. suggested format ${curr}_${stage}_${train_group}_${condition}_${fold}_${seed}')

    parser.add_argument('--script',
                           type=str,
                           default='not specified',
                           help='An identifier for the script that generated the model,'+\
                        'e.g. smoothness_v1.2')
    

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