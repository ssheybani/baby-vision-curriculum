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

from copy import deepcopy

# torchvision.disable_beta_transforms_warning()
# import torchvision.transforms.v2 as tr #May 9: would require reinstalling th evirtual env.
# we might do it later.

# import torch.nn as nn


# SCRIPT_DIR = os.path.realpath(os.path.dirname(inspect.getfile(inspect.currentframe()))) #os.getcwd() #
# # print('cwd: ',SCRIPT_DIR)
# #os.path.realpath(os.path.dirname(inspect.getfile(inspect.currentframe())))
# util_path = os.path.normpath(os.path.join(SCRIPT_DIR, '..', 'util'))
# sys.path.insert(0, util_path)    


# from train_downstream_VideoMAE import train_classifier_ddp
# from make_toybox_dataset import make_toybox_dataset
from transformers import VideoMAEConfig, VideoMAEModel
from torch.utils.data import Dataset
import av

# ------------
# Dataset and Dataloader

def _get_transform(image_size):

    mean = [0.5, 0.5, 0.5]#np.mean(mean_all, axis=0) #mean_all[chosen_subj] 
    std = [0.25, 0.25, 0.25] #std_all[chosen_subj] 
    
#     [0.485, 0.456, 0.406]  # IMAGENET_DEFAULT_MEAN
#     [0.229, 0.224, 0.225]  # IMAGENET_DEFAULT_STD

    augs = [tr.Resize(image_size), tr.CenterCrop(image_size), 
            tr.ConvertImageDtype(torch.float32), 
             tr.Normalize(mean,std)]
    return tr.Compose(augs)

def transform_vid(video):
    # Used with standard video datasets such as torchvision.UCF101
#     print(vid.shape)
    if video.shape[1]!=3: # Make it TCHW
        video = torch.permute(video, (0,3,1,2))
    image_size = 224
#     vid.p
    transform = _get_transform(image_size)
#     xtt = [transform(torch.from_numpy(frame)).unsqueeze(0) 
    xtt = [transform(frame).unsqueeze(0) 
       for frame in video]
    return torch.concat(xtt, axis=0)#.unsqueeze(0)

def transform_image(image):
#     Used for standard single image datasets such as torchvision.CIFAR10, torchvision.ImageNet
#     if image.shape[0]!=3:
    image_size=224
    num_frames=16
    transform = _get_transform(image_size)
    return transform(image).unsqueeze(0).repeat(num_frames,1,1,1)

def transform_image_cifar10(image):
#     Used for standard single image datasets such as torchvision.CIFAR10, torchvision.ImageNet
#     if image.shape[0]!=3:
    image_size=224
    num_frames=16
    mean = [0.5, 0.5, 0.5]#np.mean(mean_all, axis=0) #mean_all[chosen_subj] 
    std = [0.25, 0.25, 0.25] #std_all[chosen_subj] 
    
#     [0.485, 0.456, 0.406]  # IMAGENET_DEFAULT_MEAN
#     [0.229, 0.224, 0.225]  # IMAGENET_DEFAULT_STD

    augs = [tr.ToTensor(),
            tr.Resize(image_size), tr.CenterCrop(image_size),
            tr.ConvertImageDtype(torch.float32), 
             tr.Normalize(mean,std)]
    transform = tr.Compose(augs)

    return transform(image).unsqueeze(0).repeat(num_frames,1,1,1)


def get_inp_label(task, batch):
    if task=='ucf101':
        inputs, _, labels = batch
        return inputs, labels
    else:
        raise NotImplementedError()

def ucf_collate(batch):
    filtered_batch = []
    for video, _, label in batch:
        filtered_batch.append((video, label))
    return torch.utils.data.dataloader.default_collate(filtered_batch)

from torchvision.datasets.ucf101 import UCF101
class MyUCF101(UCF101):
    def __init__(self, root, annotation_path, frames_per_clip, step_between_clips=1, 
                 frame_rate= None, fold=1, train=True, transform=None, _precomputed_metadata=None, 
                 num_workers=1, _video_width=0, _video_height=0,
                 _video_min_dimension=0, _audio_samples=0, output_format='THWC'):
        super(MyUCF101, self).__init__(root, annotation_path, frames_per_clip, 
                                       step_between_clips=step_between_clips, 
                 frame_rate=frame_rate, fold=fold, train=train, transform=transform, 
                                       _precomputed_metadata=_precomputed_metadata, 
                 num_workers=num_workers, _video_width=_video_width, _video_height=_video_height,
                 _video_min_dimension=_video_min_dimension, _audio_samples=_audio_samples, 
                                       output_format=output_format)
        print('samples len:',len(self.samples))
        print('element type:',type(self.samples[0][0]))
        video_list = [x[0] for x in self.samples]
        
        self.indices = []
#         fold=1
        for train in [True, False]:
            fold=fold
#             for fold in range(1,4):
            self.indices+=self._select_fold(video_list, annotation_path, fold, train)
#         self.indices = self.indices[::]
        video_clips = self.full_video_clips
        self.video_clips = video_clips.subset(self.indices)
        self.transform = transform
        
def make_ucf101dataset(args):
    ucf_root='/N/project/baby_vision_curriculum/benchmarks/mainstream/ucf101/UCF-101'
    annotation_path = '/N/project/baby_vision_curriculum/benchmarks/mainstream/ucf101/UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist/'
    frames_per_clip = args.num_frames #16
    step_between_clips = 300#1
    frame_rate=args.frame_rate#int(30/args.ds_rate)
    transform = transform_vid
    output_format= 'TCHW'
    num_workers=args.num_workers-1 #40
    dataset = MyUCF101(ucf_root, 
                      annotation_path,
                      frames_per_clip,
                      step_between_clips=step_between_clips,
                      frame_rate=frame_rate,
                      fold=1,
                      train=True,
                      transform=transform,
                      output_format=output_format,
                      num_workers=num_workers)
    # Note: MyUCF101 collects both train and val split for the given fold
    num_classes = 101
    return dataset, num_classes
#     return {'train':train_dataset,
#            'val':val_dataset}, num_classes

def make_cifar10dataset(args):
    cifar10img_root = '/N/project/baby_vision_curriculum/benchmarks/mainstream/cifar10'
    image_datasets = {'train': torchvision.datasets.CIFAR10(root=cifar10img_root,
                                                            transform=transform_image_cifar10, 
                                                            train=True, download=True),
                      'val': torchvision.datasets.CIFAR10(root=cifar10img_root,
                                                          transform=transform_image_cifar10, 
                                                          train=False, download=True)}
    num_classes = 10 
        
    return image_datasets, num_classes

def make_dataset(args):
    task = args.task
    if task=='ucf101':
        return make_ucf101dataset(args)
#     seq_len = kwargs['seq_len']
#     image_size = kwargs['image_size']
    elif task=='cifar10':
        return make_cifar10dataset(args)
    else:
        raise NotImplementedError()

def get_config(image_size, args, num_labels=2):
    arch_kw = args.architecture
    
    if arch_kw=='base': #default
        config = transformers.VideoMAEConfig(image_size=image_size, patch_size=16, num_channels=3,
                                             num_frames=args.num_frames, tubelet_size=args.tubelet_size, 
                                             hidden_size=768, num_hidden_layers=12, num_attention_heads=12,
                                             intermediate_size=3072, num_labels=num_labels)
    elif arch_kw=='small2':
        hidden_size = 768
        intermediate_size = 4*768
        num_attention_heads = 6
        num_hidden_layers = 6
        
        config = transformers.VideoMAEConfig(image_size=image_size, patch_size=16, num_channels=3,
                                             num_frames=args.num_frames, tubelet_size=args.tubelet_size, 
                                             hidden_size=hidden_size, num_hidden_layers=num_hidden_layers, 
                                             num_attention_heads=num_attention_heads,
                                             intermediate_size=intermediate_size, num_labels=num_labels)
    elif arch_kw=='small1':
        hidden_size = 384
        intermediate_size = 4*384
        num_attention_heads = 6
        num_hidden_layers = 12
        
        config = transformers.VideoMAEConfig(image_size=image_size, patch_size=16, num_channels=3,
                                             num_frames=args.num_frames, tubelet_size=args.tubelet_size, 
                                             hidden_size=hidden_size, num_hidden_layers=num_hidden_layers, num_attention_heads=num_attention_heads,
                                             intermediate_size=intermediate_size, num_labels=num_labels)
        
    elif arch_kw=='small3':
        hidden_size = 384
        intermediate_size = 4*384
        num_attention_heads = 6
        num_hidden_layers = 6
        
        config = transformers.VideoMAEConfig(image_size=image_size, patch_size=16, num_channels=3,
                                             num_frames=16, tubelet_size=2, 
                                             hidden_size=hidden_size, num_hidden_layers=num_hidden_layers, num_attention_heads=num_attention_heads,
                                             intermediate_size=intermediate_size)
        
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
        for param in model.classifier.parameters():
            param.requires_grad = True
            
def get_model(image_size, num_labels, feature_extracting, args):
    config_source = get_config(image_size, args)
    model_source = transformers.VideoMAEForPreTraining(config_source)
    
    if args.init_checkpoint_path!='na':
        print('args.init_checkpoint_path:',args.init_checkpoint_path)
        # initialize the model using the checkpoint
        model_source = init_model_from_checkpoint(model_source, args.init_checkpoint_path)
  
    config_target = get_config(image_size, args, num_labels=num_labels)
    model_target = transformers.VideoMAEForVideoClassification(config=config_target)
    model_target = adapt_videomae(model_source, model_target)
    if not torch.all(
        model_target.videomae.embeddings.patch_embeddings.projection.weight==model_source.videomae.embeddings.patch_embeddings.projection.weight):
        warnings.warn('Model not successfully initialized')
    
#     if feature_extracting:
#         set_parameter_requires_grad(model_target, feature_extracting)
    
    return model_target

        
# def get_lr(optimizer):
#     for param_group in optimizer.param_groups:
#         return param_group['lr']
    
# def get_optimizer(model, feature_extract, args):
#     params_to_update = model.parameters()
#     print("Params to learn:")
#     if feature_extract:
#         params_to_update = []
#         for name,param in model.named_parameters():
#             if param.requires_grad == True:
#                 params_to_update.append(param)
#                 print("\t",name)
#     else:
#         for name,param in model.named_parameters():
#             if param.requires_grad == True:
#                 print("\t",name)

# #     if feature_extract:
#     lr = args.lr#1e-3
#     weight_decay =args.wd#5e-5
#     optimizer_ft = torch.optim.Adam(params_to_update, lr=lr, weight_decay=weight_decay)
#     #     optimizer_ft = torch.optim.SGD([{'params': params_to_update, 
#     #                               'initial_lr':lr}], 
#     #                             lr=lr, momentum=0.9)
# #     else:
# #         lr=1e-4
# #         optimizer_ft = torch.optim.Adam(params_to_update, lr=lr)
        
#     return optimizer_ft

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
    xdf['labels'] = fnames
    xdf = xdf[['labels']+ list(xdf.columns[:-1])]

#     xdf = xdf.sort_values('fnames')
#     xdf = xdf.drop_duplicates(subset='fnames', ignore_index=True)

    savedir = args.savedir
    Path(savedir).mkdir(parents=True, exist_ok=True)
#         <model vs scores>_<prot>_seed_<seed>_other_<other>_<other id>
#     result_fname = '_'.join(['embeddings', args.prot_name,           #@@@@@on May 14
#                             'seed', str(args.seed),  
#                             args.other_id])+'.csv'
    
    init_fname = Path(args.init_checkpoint_path).stem
    result_fname = 'embeddings_'+init_fname+'.csv'
    results_fpath = os.path.join(savedir, result_fname)
    xdf.to_csv(results_fpath, sep=',', float_format='%.6f', index=False)
    print('embeddings saved at ',results_fpath)

    
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
    if is_main_proc:
        print('Started compute_ucf101_embeddings')
    setup_for_distributed(is_main_proc)
    
    number_of_cpu = len(os.sched_getaffinity(0))#multiprocessing.cpu_count() #joblib.cpu_count()
    
    #Instantiate the dataset, criterion
    dataset, num_classes = make_dataset(args)
    feature_extract = True
#     model_type = 'res50'
    #-----------------
    # Create the criterion
#     criterion = torch.nn.CrossEntropyLoss()
    
    
    # Instantiate the model, optimizer
    #Load the model, adapt it to the downstream task
    image_size = 224
    feature_extract = (args.finetune=='n')
    num_classes=0 # just extract embeddings
    xmodel = get_model(image_size, num_classes, feature_extract, args)
    
    xmodel = xmodel.to(rank)
    xmodel = DDP(xmodel, device_ids=[rank], output_device=rank, 
                   find_unused_parameters=False)
    
#     scheduler = None #torch.optim.lr_scheduler.ExponentialLR(
    #     optimizer_ft,gamma=0.9, last_epoch=num_epochs)
#     optimizer = get_optimizer(xmodel, feature_extract, args) 

    # Make the dataloaders and samplers
    sampler_shuffle = True #for the distributed dampler
#     num_epochs = args.n_epoch
    batch_size = args.batch_size# 128
    pin_memory = False
    num_workers = args.num_workers #number_of_cpu-1#32
    
    if args.task=='ucf101':
        collate_fn = ucf_collate
    else:
        collate_fn = None
    print('n cpu: ', number_of_cpu, ' n workers: ', num_workers)
    
#     sampler = DistributedSampler(datasets['train'], num_replicas=world_size, 
#                                             rank=rank, shuffle=sampler_shuffle, 
#                                             seed=seed)
    sampler = DistributedSampler(dataset, num_replicas=world_size, 
                                            rank=rank, shuffle=sampler_shuffle, 
                                            seed=seed)
                     
#     dataloaders = {x: torch.utils.data.DataLoader(
#         datasets[x], batch_size=batch_size, pin_memory=pin_memory, collate_fn=collate_fn,
#         num_workers=num_workers, shuffle=False, sampler=samplers_dict[x], drop_last=True)
#                         for x in ['train', 'val']}

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

    
    print_period=10
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
#             data['fnames'].append(fnames) #may have done a shallow copy and thus incorrect
            data['fnames'].append(
                deepcopy(fnames)) #may have done a shallow copy and thus incorrect
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
            
        allfnames = np.concatenate(allfnames)
        allembeddings = np.concatenate(allembeddings)
    
        save_results(allfnames, allembeddings, args)
        

    cleanup()
    
    
#____________________________________________
if __name__ == '__main__':
    
    #---------- Parse the arguments
    parser = argparse.ArgumentParser(description='Evaluate downstream performance for a pretrained model.')

    # Add the arguments

    parser.add_argument('-task',
                           type=str,
                           default='ucf101',
                           help='')

    parser.add_argument('-init_checkpoint_path',
                           type=str,
                           help='absolute path to the checkpoint file of the pretrained model')

    parser.add_argument('-savedir',
                           type=str,
                           default='',
                           help='')
    
    parser.add_argument('--finetune',
                           type=str,
                           default='n',
                           help='y:finetune. n:linear probe')
    parser.add_argument('--frame_rate',
                           type=int,
                           default=10,
                           help='frame rate of the videos in the benchmark')
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
    parser.add_argument('--num_workers',
                           type=int,
                        default=6,
                           help='number of cpu for each gpu')
    
    parser.add_argument('--architecture',
                           type=str,
                           default='',
                           help='see get_config')    
        
    parser.add_argument('--save_model',
                           type=str,
                           default='n',
                           help='whether or not to save the model after training a classifier head')
    
    parser.add_argument('--prot_name',
                           type=str,
                        default='x',
                           help='protocol name. used only for naming the output files')

    parser.add_argument('--seed',
                           type=int,
                        default=0,
                           help='A seed used as both model ID and a random seed')


    parser.add_argument('--other_id',
                           type=str,
                           default='x',
                           help='An identifier for the checkpoint')
    
    #----------


    # Execute the parse_args() method
    args = parser.parse_args()
    
    
   #-------------
    
    n_gpu = torch.cuda.device_count()
    world_size= n_gpu

    mp.spawn(
            DDP_process,
            args=(world_size, args),#prot_arg, seed_arg),
            nprocs=world_size
        )