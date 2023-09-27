# added in v2.2:
# tube masking
# choice of adamw
# choice of small3


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


# from PIL import Image
from torch.utils.data import Dataset
import random

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

class TubeMaskingGenerator:
    def __init__(self, input_size, mask_ratio):
        self.frames, self.height, self.width = input_size
        self.num_patches_per_frame =  self.height * self.width
        self.total_patches = self.frames * self.num_patches_per_frame 
        self.num_masks_per_frame = int(mask_ratio * self.num_patches_per_frame)
        self.total_masks = self.frames * self.num_masks_per_frame

    def __repr__(self):
        repr_str = "Maks: total patches {}, mask patches {}".format(
            self.total_patches, self.total_masks
        )
        return repr_str

    def __call__(self):
        mask_per_frame = np.hstack([
            np.zeros(self.num_patches_per_frame - self.num_masks_per_frame),
            np.ones(self.num_masks_per_frame),
        ])
        np.random.shuffle(mask_per_frame)
        mask = np.tile(mask_per_frame, (self.frames,1)).flatten()
        return mask 
    
class StillVideoDataset(Dataset):
    """
    To use for video models. 
    """
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform
        self.num_frames = 16

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load the sequence of images
        chosen_fp = self.image_paths[idx][0]
        chosen_img = self.transform(
            torchvision.io.read_image(
                chosen_fp))
        return chosen_img.unsqueeze(0).repeat(self.num_frames, 1,1,1)

class ImageSequenceDataset(Dataset):
    """
    To use for video models. 
    """
    def __init__(self, image_paths, transform, shuffle=False):
        self.image_paths = image_paths
        self.transform = transform
        self.shuffle = shuffle

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load the sequence of images
        images = torch.cat([
            self.transform(torchvision.io.read_image(fp)).unsqueeze(0)
                     for fp in self.image_paths[idx]]) #with tochvision transform
#         images = [Image.open(fp) for fp in self.image_paths[idx]]
#         images = self.transform(images, return_tensors="pt").pixel_values[0] #with VideoMAEImageProcessor
        
        if self.shuffle:
            size = images.size(0)
            perm = torch.randperm(size)
            images = images[perm]
            
        return images
    
def get_train_val_split(fpathlist, val_ratio=0.1):
    """
    Splits the list of filepaths into a train list and val list
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
        assert len(fpathlist)>n_samples
        sample_stride = int(len(fpathlist)/n_samples)
        # for adult group, sample_stride ~=10. i.e. each frame contributes to more than 1 sample sequence, 
        # but doesn't appear in the same index of the sequence.

    fpathseqlist = [fpathlist[i:i+sample_len:ds_rate] 
                    for i in range(0, n_samples*sample_stride, sample_stride)]
    return fpathseqlist


def _get_transform(image_size):

    mean = [0.5, 0.5, 0.5]#np.mean(mean_all, axis=0) #mean_all[chosen_subj] 
    std = [0.25, 0.25, 0.25] #std_all[chosen_subj] 
    
#     [0.485, 0.456, 0.406]  # IMAGENET_DEFAULT_MEAN
#     [0.229, 0.224, 0.225]  # IMAGENET_DEFAULT_STD
    
    augs = [tr.Resize(image_size), tr.CenterCrop(image_size), 
            tr.ConvertImageDtype(torch.float32), 
             tr.Normalize(mean,std)]
    return tr.Compose(augs)

# def _get_transform(image_size):
# # image_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
#     image_processor = transformers.VideoMAEImageProcessor(
#         size={"shortest_edge":image_size},
#         do_center_crop=True, crop_size={"height":image_size, "width": image_size}
#     )
#     return image_processor



def make_dataset(subj_dirs, **kwargs):
    seq_len = kwargs['seq_len']
    n_groupframes=kwargs['n_groupframes']#1450000
    ds_rate = kwargs['ds_rate']
    jpg_root = kwargs['jpg_root']
    image_size = kwargs['image_size']
    fold = kwargs['fold']
    transform = _get_transform(image_size)
    gx_fpathlist = []
    for i_subj, subjdir in enumerate(tqdm(subj_dirs)):
        gx_fpathlist += get_fpathlist(jpg_root, subjdir, ds_rate=ds_rate)
    
    # added on May15
    max_folds = 3
    fold_size = int(len(gx_fpathlist)/max_folds)
    gx_fpathlist = gx_fpathlist[fold*fold_size:(fold+1)*fold_size] # for adult group
    print('Num. frames in the fold:',len(gx_fpathlist))

    #     if len(gx_fpathlist)>=n_groupframes:
#         gx_fpathlist = gx_fpathlist[:n_groupframes]
#         # 1450000/16 = 90625 => n_trainsamples=81560, n_valsamples= 9060
#         # 1274 iterations of train. 141 iterations of test. 
#         n_trainsamples = None
#         n_valsamples = None
#     else:
    
    n_trainsamples = int(0.9*n_groupframes/seq_len) #81k
    n_valsamples = None #means don't do bootstraping for val. Use whatever number 0.1*len(gx_fpathlist) gives.

    # Train-val split
    gx_train_fp, gx_val_fp = get_train_val_split(gx_fpathlist, val_ratio=0.1)


    gx_train_fpathseqlist = get_fpathseqlist(gx_train_fp, seq_len, ds_rate=1, n_samples=n_trainsamples)
    gx_val_fpathseqlist = get_fpathseqlist(gx_val_fp, seq_len, ds_rate=1, n_samples=n_valsamples)
    
    return {'train':ImageSequenceDataset(gx_train_fpathseqlist, transform=transform, shuffle=True),
           'val': ImageSequenceDataset(gx_val_fpathseqlist, transform=transform)}

def get_config(image_size, args):
    arch_kw = args.architecture
    
    if arch_kw=='base': #default
        config = transformers.VideoMAEConfig(image_size=image_size, patch_size=16, num_channels=3,
                                             num_frames=16, tubelet_size=2, 
                                             hidden_size=768, num_hidden_layers=12, num_attention_heads=12,
                                             intermediate_size=3072, initializer_range=0.02,
                                             use_mean_pooling=True, decoder_num_attention_heads=6,
                                             decoder_hidden_size=384, decoder_num_hidden_layers=4, 
                                             decoder_intermediate_size=1536, norm_pix_loss=True)
    elif arch_kw=='small1':
        hidden_size = 384
        intermediate_size = 4*384
        num_attention_heads = 6
        num_hidden_layers = 12
        
        config = transformers.VideoMAEConfig(image_size=image_size, patch_size=16, num_channels=3,
                                             num_frames=16, tubelet_size=2, 
                                             hidden_size=hidden_size, num_hidden_layers=num_hidden_layers, num_attention_heads=num_attention_heads,
                                             intermediate_size=intermediate_size)
        
    elif arch_kw=='small2':
        hidden_size = 768
        intermediate_size = 4*768
        num_attention_heads = 6
        num_hidden_layers = 6
        
        config = transformers.VideoMAEConfig(image_size=image_size, patch_size=16, num_channels=3,
                                             num_frames=16, tubelet_size=2, 
                                             hidden_size=hidden_size, num_hidden_layers=num_hidden_layers, num_attention_heads=num_attention_heads,
                                             intermediate_size=intermediate_size)
        
    elif arch_kw=='small3':
        hidden_size = 384
        intermediate_size = 4*384
        num_attention_heads = 6
        num_hidden_layers = 6
        
        config = transformers.VideoMAEConfig(image_size=image_size, patch_size=16, num_channels=3,
                                             num_frames=16, tubelet_size=2, 
                                             hidden_size=hidden_size, num_hidden_layers=num_hidden_layers, num_attention_heads=num_attention_heads,
                                             intermediate_size=intermediate_size)
        
    
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

def get_optim(args):
    if args.optim=='SGD':
        return 
    
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

    
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
    seed = args.data_seed
#     n_epoch = args.n_epoch
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
    
    print('OPENBLAS_NUM_THREADS: ', os.environ['OPENBLAS_NUM_THREADS'])
    
    other_seed = args.other_seed #np.random.randint(1000)
    torch.manual_seed(other_seed)
    random.seed(args.data_seed)
    
    number_of_cpu = len(os.sched_getaffinity(0))#multiprocessing.cpu_count() #joblib.cpu_count()
    

    
    # Instantiate the model, optimizer
    # xmodel = get_model('res50')
    image_size = 224
    
    xmodel = get_model(image_size, args)
    
    if args.init_checkpoint_path!='na':
        print('args.init_checkpoint_path:',args.init_checkpoint_path)
        # initialize the model using the checkpoint
        xmodel = init_model_from_checkpoint(xmodel, args.init_checkpoint_path)
        
    seq_len = xmodel.config.num_frames #equivalent to num_frames in VideoMAE()
    num_patches_per_frame = (xmodel.config.image_size // xmodel.config.patch_size) ** 2
    model_seq_length = (xmodel.config.num_frames // xmodel.config.tubelet_size) * num_patches_per_frame
    mask_ratio = args.mask_ratio#float(args.mask_ratio)/100 #0.9 #@@@ switch to float if possible
    # num_masks = int(mask_ratio * model_seq_length)
    
    nseq_time = xmodel.config.num_frames // xmodel.config.tubelet_size
    nseq_space = xmodel.config.image_size // xmodel.config.patch_size
    
    xmodel = xmodel.to(rank)
    # print("model device", xmodel.device)
    xmodel = DDP(xmodel, device_ids=[rank], output_device=rank, 
                   find_unused_parameters=False)
    
    lr=args.lr#1e-3 #1e-2#5e-3#1e-4
    wd =args.wd #5e-5 # 1e-4
    momentum=args.momentum
    
    if args.optim=='sgd':
        optimizer = torch.optim.SGD(xmodel.parameters(), lr=lr, weight_decay=wd, 
                                    momentum=momentum, nesterov=True)
    elif args.optim=='adamw':
        optimizer = torch.optim.AdamW(xmodel.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.95))
    else:
        optimizer = torch.optim.Adam(xmodel.parameters(), lr=lr, weight_decay=wd)

    sampler_shuffle = True #@@@@ True #for the distributed dampler
    # Set the other hyperparams: n_epoch, batch_size, num_workers
    num_epochs = args.n_epoch #per train stage
    batch_size = args.batch_size #16 #128 #For individual GPUs
    pin_memory = False#True
    num_workers = 6#2#int((number_of_cpu-1)/4) #2 #0#1#2#3 #
#     prefetch_factor = None #int(1.5*batch_size/num_workers)

    print('mask_ratio:', mask_ratio, 'lr:', lr) #@@@ Debug
    
    print('n cpu: ', number_of_cpu, ' n workers: ', num_workers)
    
    print('Model, optimizer, etc instantiated')
    
    print('seed: ',seed)
    # Split the data
    
    jpg_root=args.jpg_root #'/N/project/infant_image_statistics/preproc_saber/JPG_10fps/'
    ds_rate = args.ds_rate #1

    n_groupframes = 1450000 # minimum number of frames across age groups
    g0='008MS+009SS+010BF+011EA+012TT+013LS+014SN+015JM+016TF+017EW'
    g1='026AR+027SS+028CK+028MR+029TT+030FD+031HW+032SR+033SE+034JC'
    g2='043MP+044ET+046TE+047MS+048KG+049JC+050AB+050AK+051DW'
    g3='BR+CW+EA+ED+JB+KI+LS+SB+TR'
#     g0='008MS+009SS_withrotation+010BF_withrotation+011EA_withrotation+012TT_withrotation+013LS+014SN+015JM+016TF+017EW_withrotation'
#     g1='026AR+027SS+028CK+028MR+029TT+030FD+031HW+032SR+033SE+034JC_withlighting'
#     g2='043MP+044ET+046TE+047MS+048KG+049JC+050AB+050AK_rotation+051DW'
    
# Total number of frames in each age group: g0=1.68m, g1=1.77m, g2=1.45m

    
    g0 = g0.split('+')
    g1 = g1.split('+')
    g2 = g2.split('+')
    g3 = g3.split('+')
    
    gRand=[]
    for gx in [g0,g1,g2,g3]:
        gRand.extend(random.sample(gx, 3))
    random.shuffle(gRand)
    
    group_dict = {"g0":g0, "g1":g1, "g2":g2, "g3":g3, 'gr':gRand}
    group = group_dict.get(args.train_group)
    print(group)                                               
    # for i_p, fname_part in enumerate(fname_parts):
        # perform the training
        # ideally could be replaced with a line like this:
            # model, results = train_ddp(model, fnames, optimizer, criterion, 
            #     n_ep, world_size, rank)
        
        # make the dataset, sampler and dataloader
        # dataset = SimpleImageDataset(fname_part, transform_contrastive_hv)
    # dataset = SequentialHomeviewDataset(fname_part, n_views, t_neigh, 
                                            # transform=transform_homeview)
    
    #     ds_rate = 1
    # n_samples = None#10 #50000
    datasets = make_dataset(group, seq_len=seq_len, jpg_root=jpg_root, ds_rate=ds_rate, n_groupframes=n_groupframes, 
                            fold=args.fold, image_size=image_size)
    print('datasets[train][0].shape:',datasets['train'][0].shape) #@@@ debugging
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
        drop_last=True)
                        for x in ['train', 'val']}

    
    
    maskgen_input_size = nseq_time, nseq_space, nseq_space
    mask_ratio = 0.9
    num_patches_per_frame
    mask_gen = TubeMaskingGenerator(maskgen_input_size, mask_ratio)
    print('len dset, len dloader: ', len(datasets['train']), len(dataloaders['train']))
#         print(dataset.__getitem__(22).shape)
    print('dataloaders created') #@@@
        
    verbose = (verbose and is_main_process())
    
    
    train_loss_history = []
    val_loss_history = []
    
    best_loss = float('inf')
    
#     STAGE = 1
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
            
            i_iter,print_interval =0,10
#             i_break, print_interval = 10,1 #@@@@@@@@@@@@@ debugging
            for inputs in tqdm(dataloaders[phase]):
#                 print(inputs.shape)
                optimizer.zero_grad()

                # Shuffle each array in the batch
                bool_masked = np.zeros((batch_size, model_seq_length))
                for i_el in range(batch_size):#bool_masked.shape[0]):
                    bool_masked[i_el,:] = mask_gen()
                    # np.random.shuffle(bool_masked[i])
                bool_masked_pos = torch.from_numpy(bool_masked).bool()
                bool_masked_pos = bool_masked_pos.to(rank)
                # print("bool_masked_pos device", bool_masked_pos.get_device())
                inputs = inputs.to(rank)
                # print("input device", inputs.get_device())
                # bool_masked_pos = torch.randint(0, 2, (batch_size, model_seq_length)).bool()
                outputs = xmodel(inputs, bool_masked_pos=bool_masked_pos)
                loss = outputs.loss
                
                if i_iter%print_interval==0:
                    print('loss:',loss.item(), flush=True)
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                
                i_iter+=1
#                 if (i_iter==i_break) | (phase=='val'):
#                     break #@@@

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
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        results_fpath = os.path.join(model_dir,f"train_val_scores_{train_group}_seed_{seed}"+args.other_id)
        results_df.to_csv(results_fpath+'.csv', sep=',', float_format='%.4f')
        # the model
        # STAGE = "g2"
        model_fname = '_'.join(['model', train_group, 'seed',str(seed), 'other', str(other_seed), args.other_id])+'.pt'
        MODELPATH = os.path.join(model_dir,model_fname)
        SCRIPT = script_arg
        # TRAIN_SETS = str(subjnames)

        torch.save({
                'init_checkpoint_path':args.init_checkpoint_path,
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
    parser.add_argument('-train_group',
                           type=str,
                           help='The age group on which the model gets trained. g0 or g1 or g2 or rand')

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
                           default='adam',
                           help='')
    
    parser.add_argument('--lr',
                           type=float,
                           default=1e-3,
                           help='')
    parser.add_argument('--wd',
                           type=float,
                           default=5e-5,
                           help='')
    parser.add_argument('--momentum',
                           type=float,
                           default=0.9,
                           help='')
    parser.add_argument('--batch_size',
                           type=int,
                           default=16,
                           help='')
        
    parser.add_argument('--architecture',
                           type=str,
                           default='',
                           help='see get_config')    
        
    parser.add_argument('--n_epoch',
                           type=int,
                           default=2,
                           help='')
    
    parser.add_argument('--data_seed',
                           type=int,
                        default=0,
                           help='')
    
    parser.add_argument('--other_seed',
                           type=int,
                        default=0,
                           help='A seed used as both model ID and a random seed')


    parser.add_argument('--other_id',
                           type=str,
                           default='',
                           help='An identifier for the checkpoint')

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