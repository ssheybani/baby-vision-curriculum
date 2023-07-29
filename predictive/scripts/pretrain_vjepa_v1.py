# adapted from official IJEPA repo

#added in v1:
# Each sample has two shorter sequences instead of one long sequence. The two samples 
# come from separate parts of the videos, determined by args.interval
# The first item in the sample will be used by the encoder and the second item by the decoder
# TwoSeqDataset
# the ability to include additional data augmentations.

import sys, os

env_root = '/N/project/baby_vision_curriculum/pythonenvs/hfenv/lib/python3.10/site-packages/'
sys.path.insert(0, env_root)

# SCRIPT_DIR = os.path.realpath(os.path.dirname(inspect.getfile(inspect.currentframe()))) #os.getcwd() #
# print('cwd: ',SCRIPT_DIR)
#os.path.realpath(os.path.dirname(inspect.getfile(inspect.currentframe())))
# util_path = os.path.normpath(os.path.join(SCRIPT_DIR, '..', 'util'))
# sys.path.insert(0, util_path)
os.environ['OPENBLAS_NUM_THREADS'] = '38' #@@@@ to help with the num_workers issue
os.environ['OMP_NUM_THREADS'] = '1'  #10


import numpy as np
import torch#, torchvision
# from torchvision import transforms as tr
from tqdm import tqdm
from pathlib import Path
# import math
import argparse
# import pandas as pd
# import warnings


import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp



# from PIL import Image
# from torch.utils.data import Dataset
import random
from copy import deepcopy

# device_index = 0  # Use 0 for the first GPU, 1 for the second, and so on
# Get the GPU properties

from ddputils import is_main_process, save_on_master, setup_for_distributed
from homeview import (_get_transform, get_fpathlist,
    get_fold, get_train_val_split, get_fpath2framelist, 
    TwoFrameDataset, get_group)
from mask import MaskCollator as MBMaskCollator
import vision_transformer as vit
from tensors import trunc_normal_
from helper import init_opt, load_checkpoint
from mask import update_masks, apply_masks
from loggingtools import (
        CSVLogger,
        grad_logger,
        AverageMeter,
        gpu_timer)
import yaml
import torch.nn.functional as F
from tensors import repeat_interleave_batch
from distributed import AllReduce

import logging

        
        
def make_dataset(subj_dirs, image_size, args):
    seq_len = args.num_frames #kwargs['seq_len']
#     n_groupframes=kwargs['n_groupframes']#1450000
    ds_rate = args.ds_rate #kwargs['ds_rate']
    jpg_root = args.jpg_root #kwargs['jpg_root']
#     image_size = kwargs['image_size']
    fold = args.fold #kwargs['fold']
    condition = args.condition #kwargs['condition']
    n_trainsamples = args.n_trainsamples
    augs = args.augs
    crop_scale = (0.3,1.)
    
    transform = _get_transform(image_size, augs=augs, crop_size=image_size, crop_scale=crop_scale)
    gx_fpathlist = []
    for i_subj, subjdir in enumerate(tqdm(subj_dirs)):
        gx_fpathlist += get_fpathlist(jpg_root, subjdir, ds_rate=ds_rate)
    
    max_folds = 3
    gx_fpathlist = get_fold(gx_fpathlist, fold, max_folds, args)
    print('Num. frames in the fold:',len(gx_fpathlist))

    if condition=='shuffle':
        random.shuffle(gx_fpathlist)
    
    
    # elif condition=='static':
    #     train_dataset = StillVideoDataset(gx_train_fpathseqlist, transform=transform)
    #     val_dataset = ImageSequenceDataset(gx_val_fpathseqlist, transform=transform, shuffle=False)
#         StillVideoDataset(gx_val_fpathseqlist, transform=transform)

    
    gx_fpath2framelist = get_fpath2framelist(gx_fpathlist, args.interval, n_samples=n_trainsamples)
    train_dataset = TwoFrameDataset(gx_fpath2framelist, transform=transform)
#         val_dataset = TwoFrameDataset(gx_val_fpathseqlist, transform=transform, shuffle=False)
        
    return {'train':train_dataset,
           'val': None}

def get_model(
    device,
    patch_size=16,
    tubelet_size=1,
    num_frames=1,
    model_name='vit_base',
    image_size=224,
    pred_depth=6,
    pred_emb_dim=384
):
#     encoder = vit_base(
    encoder = vit.__dict__[model_name](
        img_size=[image_size],
        patch_size=patch_size,
        num_frames=num_frames,
        tubelet_size=tubelet_size) #@@@
#     predictor = vit_predictor(
    predictor = vit.__dict__['vit_predictor'](
        sequence_shape=encoder.sequence_shape,
        embed_dim=encoder.embed_dim,
        predictor_embed_dim=pred_emb_dim,
        depth=pred_depth,
        num_heads=encoder.num_heads)

    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)

    for m in encoder.modules():
        init_weights(m)

    for m in predictor.modules():
        init_weights(m)

    encoder.to(device)
    predictor.to(device)
    # logger.info(encoder)
    return encoder, predictor

def save_checkpoint(chpt_path,
                    encoder, predictor, target_encoder,
                    epoch, loss_meter, batch_size, world_size, lr,
                    optimizer, scaler):
    save_dict = {
        'encoder': encoder.module.state_dict(),
        'predictor': predictor.module.state_dict(),
        'target_encoder': target_encoder.module.state_dict(),
        'opt': optimizer.state_dict(),
        'scaler': None if scaler is None else scaler.state_dict(),
        'epoch': epoch,
        'loss': loss_meter.avg,
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
#     n_epoch = args.n_epoch
    # script_arg = args.script

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
    print('is_main_proc', is_main_proc) #@@@@
    setup_for_distributed(is_main_proc) #removes printing for child processes.
    
    print('OPENBLAS_NUM_THREADS: ', os.environ['OPENBLAS_NUM_THREADS'])
    
    # other_seed = args.other_seed #np.random.randint(1000)
    other_seed = args.seed
    torch.manual_seed(other_seed)
    random.seed(args.seed)
    
    number_of_cpu = len(os.sched_getaffinity(0))#multiprocessing.cpu_count() #joblib.cpu_count()
    
    image_size = 224
    patch_size=16
    pms1 = args.pred_mask_scale
    pred_mask_scale=(pms1, pms1+0.05)
    ems1 = args.enc_mask_scale
    enc_mask_scale=(ems1,ems1+0.15)#(0.85,1)
    aspect_ratio=(0.75,1.5)
    num_enc_masks=1
    num_pred_masks=4
    allow_overlap=(args.allow_overlap=='y')
    min_keep=10


    verbose = (verbose and is_main_process())
    
    # -- LOGGING
    folder = args.savedir #'/N/project/baby_vision_curriculum/trained_models/predictive/v0/jul17/'
    Path(folder).mkdir(parents=True, exist_ok=True)
    #args['logging']['folder']
    # tag = args.other_id #'jepa'
    #args['logging']['write_tag']
    
    params_file = 'params_'+args.run_id+'.yaml'
    dump = os.path.join(folder, params_file)
    with open(dump, 'w') as f:
        yaml.dump(args, f)
    
    # -- log/checkpointing paths
    
    
    
    #         results_df.to_csv(results_fpath+'.csv', sep=',', float_format='%.4f')
            
    #         model_fname = '_'.join(['model', train_group, 'seed',str(seed), 'other', str(other_seed), args.other_id])+'.pt'
    #         MODELPATH = os.path.join(model_dir,model_fname)
    
    log_path = os.path.join(folder, 'csvlog_'+args.run_id+'.csv')
    # log_path = os.path.join(folder, f'{tag}_r{rank}.csv')
    # chpt_path = os.path.join(folder, f'{tag}' + '-ep{epoch}.pth.tar')
    chpt_fname = 'model_'+args.run_id+'.pth.tar'
    chpt_path = os.path.join(folder, chpt_fname)
    # latest_path = os.path.join(folder, f'{tag}-latest.pth.tar') #save checkpoint path
    # load_path = None
    # r_file = None
    # load_model = (args.init_checkpoint_path!='na')
    load_path = args.init_checkpoint_path
    # if load_model:
    #     load_path = os.path.join(folder, r_file) if r_file is not None else latest_path

    # -- make csv_logger
    csv_logger = CSVLogger(log_path,
                           ('%d', 'epoch'),
                           ('%d', 'itr'),
                           ('%.5f', 'loss'),
                           ('%.4e', 'grad-FL'),
                           ('%.4e', 'grad-LL'),
                           ('%d', 'mask-A'),
                           ('%d', 'mask-B'),
                           ('%d', 'time (ms)'))
    
    mask_collator = MBMaskCollator(
        input_size=image_size,
        patch_size=patch_size,
        pred_mask_scale=pred_mask_scale,
        enc_mask_scale=enc_mask_scale,
        aspect_ratio=aspect_ratio,
        nenc=num_enc_masks,
        npred=num_pred_masks,
        allow_overlap=allow_overlap,
        min_keep=min_keep)
    
    
    # -- init model
    
    
    device= rank#'cuda:0'#'cpu'
    model_name='vit_'+args.architecture #'vit_base'#'vit_small'
    pred_depth=6
    pred_emb_dim=384
    patch_size = 16
    tubelet_size=args.tubelet_size#2#1 #@@@
    num_frames = args.num_frames #6#6#1 #@@@

    encoder, predictor = get_model(
        device=device,
        patch_size=patch_size,
        image_size=image_size,
        pred_depth=pred_depth,
        pred_emb_dim=pred_emb_dim,
        model_name=model_name,
        tubelet_size=tubelet_size,
        num_frames=num_frames)
    target_encoder = deepcopy(encoder)
    

        
        
    

    
    lr=args.lr#1e-3 #1e-2#5e-3#1e-4
    wd =args.wd #5e-5 # 1e-4
    momentum=args.momentum
    ipe=args.max_epoch_iters #iterations_per_epoch ~2000?
    use_bfloat16=True
    num_epochs=args.n_epoch
    ipe_scale=1
    
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        encoder=encoder,
        predictor=predictor,
        wd=wd,
        final_wd=wd,
        start_lr=lr,
        ref_lr=lr,
        momentum=momentum,
        final_lr=lr,
        iterations_per_epoch=ipe,
        warmup=0,
        num_epochs=num_epochs,
        ipe_scale=ipe_scale,
        use_bfloat16=use_bfloat16)
    #     xmodel = DDP(xmodel, device_ids=[rank], output_device=rank, 
#                    find_unused_parameters=False)
    load_path = args.init_checkpoint_path
    
    if load_path!='na':
        print('load_path:',load_path)
        encoder, predictor, target_encoder, optimizer, scaler, start_epoch = load_checkpoint(
            r_path=load_path,
            encoder=encoder,
            predictor=predictor,
            target_encoder=target_encoder,
            opt=optimizer,
            scaler=scaler)
        for _ in range(start_epoch*ipe):
            mask_collator.step()
            
    encoder = DDP(encoder, static_graph=True)
    predictor = DDP(predictor, static_graph=True)
    target_encoder = DDP(target_encoder)
    for p in target_encoder.parameters():
        p.requires_grad = False
    
    # -- momentum schedule
    ema = (0.996, 1.0)
    momentum_scheduler = (ema[0] + i*(ema[1]-ema[0])/(ipe*num_epochs*ipe_scale)
                          for i in range(int(ipe*num_epochs*ipe_scale)+1))


    sampler_shuffle = True #@@@@ True #for the distributed dampler
    # Set the other hyperparams: n_epoch, batch_size, num_workers
    num_epochs = args.n_epoch #per train stage
    batch_size = args.batch_size #16 #128 #For individual GPUs
    pin_memory = False#True
    num_workers = 6#2#int((number_of_cpu-1)/4) #2 #0#1#2#3 #
#     prefetch_factor = None #int(1.5*batch_size/num_workers)

    print('n cpu: ', number_of_cpu, ' n workers: ', num_workers)
    
    print('Model, optimizer, etc instantiated')
    
    print('seed: ',seed)
    # Split the data
    
    # jpg_root=args.jpg_root #'/N/project/infant_image_statistics/preproc_saber/JPG_10fps/'
    # ds_rate = args.ds_rate #1

    group = get_group(train_group)
    print(group)                                               
    datasets = make_dataset(group, image_size, args)
    
    print('datasets[train][0].shape:',datasets['train'][0].shape) #@@@ debugging
    # sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, 
                                     # shuffle=sampler_shuffle, seed=seed)

    
    samplers_dict = {x: DistributedSampler(datasets[x], num_replicas=world_size, 
                                           rank=rank, shuffle=sampler_shuffle, 
                                           seed=seed)
                     for x in ['train']}#, 'val']}

    dataloaders = {x: torch.utils.data.DataLoader(
        datasets[x], batch_size=batch_size, pin_memory=pin_memory, 
        num_workers=num_workers, shuffle=False, sampler=samplers_dict[x],
        drop_last=True, collate_fn=mask_collator)
                        for x in ['train']}#, 'val']}

    
    print('len dset, len dloader: ', len(datasets['train']), len(dataloaders['train']))
#         print(dataset.__getitem__(22).shape)
    print('dataloaders created') #@@@
        

    

            # if (epoch + 1) % checkpoint_freq == 0:
            #     torch.save(save_dict, chpt_path.format(epoch=f'{epoch + 1}'))
            

    # loss_record_period = args.loss_record_period
#     train_loss_history = []
    # val_loss_history = []
    start_epoch = 0

    # --
    # log_timings = True
    log_freq = 10
    # checkpoint_freq = 50
    # --
    
    
    phase = 'train'
#     for i_ep in range(num_epochs):
    for epoch in range(start_epoch, start_epoch+num_epochs):
        if verbose:
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)
        # Each epoch has a training and validation phase
        
        loss_meter = AverageMeter()
        maskA_meter = AverageMeter()
        maskB_meter = AverageMeter()
        time_meter = AverageMeter()

        for itr, (udata, masks_enc, masks_pred) in enumerate(
            tqdm(dataloaders[phase])):

            if itr>args.max_epoch_iters:
                break #@@@
            masks_enc = update_masks(masks_enc, 
                                     image_size, patch_size, num_frames, tubelet_size,
                                     isencoder=True)
            masks_pred = update_masks(masks_pred, 
                                      image_size, patch_size, num_frames, tubelet_size,
                                      isencoder=False)

            def load_imgs():
                # -- unsupervised imgs
                imgs = udata.to(device, non_blocking=True)
                masks_1 = [u.to(device, non_blocking=True) for u in masks_enc]
                masks_2 = [u.to(device, non_blocking=True) for u in masks_pred]
                return (imgs, masks_1, masks_2)
            imgs, masks_enc, masks_pred = load_imgs()
            maskA_meter.update(len(masks_enc[0][0]))
            maskB_meter.update(len(masks_pred[0][0]))

            def train_step():
    #             _new_lr = scheduler.step()
    #             _new_wd = wd_scheduler.step()
                # --

                def forward_target():
                    with torch.no_grad():
                        h = target_encoder(imgs)
                        h = F.layer_norm(h, (h.size(-1),))  # normalize over feature-dim
                        B = len(h)
                        # -- create targets (masked regions of h)
                        h = apply_masks(h, masks_pred)
    #                     print('h shape:', h.shape)
                        h = repeat_interleave_batch(h, B, repeat=len(masks_enc))
                        return h

                def forward_context():
                    z = encoder(imgs, masks_enc)
                    z = predictor(z, masks_enc, masks_pred)
                    return z

                def loss_fn(z, h):
                    loss = F.smooth_l1_loss(z, h)
                    loss = AllReduce.apply(loss)
                    return loss

                # Step 1. Forward
                with torch.cuda.amp.autocast(dtype=torch.bfloat16, 
                                             enabled=use_bfloat16):
                    h = forward_target()
                    z = forward_context()
                    loss = loss_fn(z, h)

                #  Step 2. Backward & step
                if phase == 'val':
                    return float(loss),0

                if use_bfloat16:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                # grad_mean = grad_logger(encoder.named_parameters(),
                #                         op='mean')
                # grad_std = grad_logger(encoder.named_parameters(),
                #                         op='std')
                grad_stats = grad_logger(encoder.named_parameters())#(grad_mean, grad_std)#
                optimizer.zero_grad()

                # Step 3. momentum update of target encoder
                with torch.no_grad():
                    m = next(momentum_scheduler)
                    for param_q, param_k in zip(encoder.parameters(), target_encoder.parameters()):
                        param_k.data.mul_(m).add_((1.-m) * param_q.detach().data)
                return (float(loss), grad_stats)

            if phase=='val':
                (loss, _), etime = gpu_timer(train_step)
                logger.info('val loss: %.3f etime %.1f ms', loss, etime)
                continue
            (loss, grad_stats), etime = gpu_timer(train_step)
            loss_meter.update(loss)
            time_meter.update(etime)

            # -- Logging
            def log_stats():
                # grad_mean,grad_std = grad_stats
                # csv_logger.log(epoch + 1, itr, loss, 
                #                grad_mean.first_layer,
                #                grad_mean.last_layer,
                #                grad_std.first_layer,
                #                grad_std.last_layer,
                #                maskA_meter.val, maskB_meter.val, etime)
                csv_logger.log(epoch + 1, itr, loss, 
                               grad_stats.first_layer,
                               grad_stats.last_layer,
                               maskA_meter.val, maskB_meter.val, etime)
                if (itr % log_freq == 0) or np.isnan(loss) or np.isinf(loss):
                    print('[%d, %5d] loss: %.3f '
                                'masks: %.1f %.1f '
                                '[mem: %.2e] '
                                '(%.1f ms)'
                                % (epoch + 1, itr,
                                   loss_meter.avg,
                                   maskA_meter.avg,
                                   maskB_meter.avg,
                                   torch.cuda.max_memory_allocated() / 1024.**2,
                                   time_meter.avg))

                    if grad_stats is not None:
                        logger.info('[%d, %5d] grad_stats: [%.2e %.2e]'
                                    % (epoch + 1, itr,
                                       grad_stats.first_layer,
                                       grad_stats.last_layer,))

            log_stats()

            assert not np.isnan(loss), 'loss is nan'

        # -- Save Checkpoint after every epoch
        logger.info('avg. loss %.3f' % loss_meter.avg)
        dist.barrier() #@@@ synchronize all ranks. to avoid thread race condition at the beginning of the next epoch or when saving the model.
#         dist.all_reduce(running_loss, op=dist.ReduceOp.SUM, async_op=False)
            
    if verbose:
        print('Training complete')
#         print('Best val loss: {:4f}'.format(best_loss.item()))

#     results_df = pd.DataFrame({'epoch': np.arange(len(train_loss_history)),
#                                'train_loss':train_loss_history,
#                                'val_loss': val_loss_history})

    if is_main_process():
#         save_checkpoint(epoch+1)
        save_checkpoint(chpt_path,
                    encoder, predictor, target_encoder,
                    epoch+1, loss_meter, batch_size, world_size, lr,
                    optimizer, scaler)
        
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
        
    parser.add_argument('--enc_mask_scale',
                           type=float,
                           default=0.85,
                           help='')
    parser.add_argument('--pred_mask_scale',
                           type=float,
                           default=0.15,
                           help='')
    parser.add_argument('--allow_overlap',
                           type=str,
                           default='y',
                           help='allow overlap between the context and the targets')
        
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
    parser.add_argument('--num_frames',
                           type=int,
                           default=16,
                           help='16 or 32')
    parser.add_argument('--tubelet_size',
                           type=int,
                           default=2,
                           help='temporal size of each patch')
    parser.add_argument('--interval',
                           type=int,
                           default=30,
                           help='interval between anchor and positive in the unit of steps (depends on ds_rate)')    
    parser.add_argument('--augs',
                           type=str,
                           default='n',
                           help='c: RandomResizedCrop, b:GaussianBlur, j:ColorJitter')  
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
                args=(world_size, args),#prot_arg, seed_arg),
                nprocs=world_size
            )
    except:
        cleanup()