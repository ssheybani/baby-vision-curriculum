"""
Report downstream performance scores for a pretrained model.
"""

import sys, os, inspect

if 'BigRed200' in os.getcwd().split('/'):
#     print('Running on BigRed200')
    sys.path.insert(0,'/N/slate/hhansar/hfenv/lib/python3.10/site-packages')
    
import argparse
import numpy as np
from pathlib import Path
import torch, torchvision
import torchvision.transforms as tr
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.nn as nn

SCRIPT_DIR = os.path.realpath(os.path.dirname(inspect.getfile(inspect.currentframe()))) #os.getcwd() #
# print('cwd: ',SCRIPT_DIR)
#os.path.realpath(os.path.dirname(inspect.getfile(inspect.currentframe())))
util_path = os.path.normpath(os.path.join(SCRIPT_DIR, '..', 'util'))
sys.path.insert(0, util_path)    

from ddputils import is_main_process, save_on_master, setup_for_distributed

from train_downstream_VideoMAE import train_classifier_ddp
# from make_toybox_dataset import make_toybox_dataset
from transformers import VideoMAEConfig, VideoMAEModel

# ------------
# Dataset and Dataloader

def _get_transform(task):
    mean = [0.5, 0.5, 0.5]#np.mean(mean_all, axis=0) #mean_all[chosen_subj] 
    std = [0.25, 0.25, 0.25] #std_all[chosen_subj] 

    if task=='toybox':
        return tr.Compose([
            tr.Resize(224),
            tr.CenterCrop(224),
            tr.ConvertImageDtype(torch.float32),
            tr.Normalize(mean,std)]
            )
    
    elif task=='cifar10':
        return tr.Compose([
            # tr.Resize(224),
            # tr.CenterCrop(224),
            tr.ToTensor()]
            # tr.ConvertImageDtype(torch.float32),
            # tr.Normalize(mean,std)]
            )

    elif task=='stl10':
        return tr.Compose([
            tr.Resize(224),
            tr.CenterCrop(224),
            tr.ToTensor(),
            tr.ConvertImageDtype(torch.float32),
            tr.Normalize(mean,std)]
            )
    else:
        raise ValueError
        

def make_dataset(task):
    transform = _get_transform(task)
    
    # if task=='toybox':
    #     toyboximg_root = '/N/scratch/sheybani/toybox_img/'
    #     image_datasets = make_toybox_dataset(toyboximg_root, transform, split='exemplar')
    #     num_classes = 12 #397 #

    if task=='cifar10':
        cifar10img_root = '/N/slate/hhansar/cifar10/'
        image_datasets = {'train': torchvision.datasets.CIFAR10(root=cifar10img_root,transform=transform, train=True, download=True),
                          'val': torchvision.datasets.CIFAR10(root=cifar10img_root,transform=transform, train=False, download=True)}
        num_classes = 10 
     
    # elif task=='stl10':
    #     stl10_root = '/N/scratch/sheybani/stl10/'
    #     image_datasets = {'train': torchvision.datasets.STL10(stl10_root, split='train', folds=None, 
    #                               transform=_get_transform(task), download=True),
    #                       'val': torchvision.datasets.STL10(stl10_root, split='test', folds=None, 
    #                               transform=_get_transform(task), download=True),
    #            }
    #     num_classes = 10 
    
    else:
        raise ValueError
        
    return image_datasets, num_classes


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

# def _adapt_model_classification(model, n_features, n_readout):
#     model.fc = torch.nn.Linear(n_features, n_readout)
#     _ = model.float()
#     return model

# def _adapt_model_simclr(model, n_features, n_out): 
#     model.fc = torch.nn.Sequential(
#         torch.nn.Linear(n_features, n_out), 
#         torch.nn.ReLU(), 
#         torch.nn.Linear(n_out, n_out)) 
#     _ = model.float()
#     return model
def _adapt_model_downstream(model, n_features, n_readout):
    model.fc = torch.nn.Linear(n_features, n_readout)
    _ = model.float()

    return model

def get_model(model_fpath, num_classes, device, model_type='res50',
              feature_extract=True):       
    # if model_type != 'res50':
    #     raise NotImplementedError
    
    # in_pretrained = False
    # if model_fpath == 'imagenet':
    #     in_pretrained = True
    # xmodel = torchvision.models.resnet50(pretrained=in_pretrained)
    # n_features, n_out = 2048, 2048
    # xmodel = _adapt_model_simclr(xmodel, n_features, n_out)
    
    # if model_fpath == 'random':
    #     pass
    # elif model_fpath == 'imagenet':
    #     pass
    # else:
    #     xload = torch.load(model_fpath, map_location=torch.device(device))#
    #     xmodel.load_state_dict(xload['model_state_dict'])
    #     del xload
        
    #  #set to False if retraining the while network
    class CIFAR10Benchmark(nn.Module):
        def __init__(self, backbone, num_classes):
            super().__init__()
    #         self.flatten = nn.Flatten()
            self.backbone = backbone
            self.classifier = nn.Linear(backbone.config.hidden_size, num_classes)

        def forward(self, x):
    #         x = self.flatten(x)
            outputs = self.backbone(x)
            pooled_output = outputs.last_hidden_state[:, 0]
            logits = self.classifier(pooled_output)
            return logits

    config = VideoMAEConfig(hidden_size = 384, num_hidden_layers = 12, num_attention_heads = 6)
    xmodel = VideoMAEModel(config)
    xload = torch.load(model_fpath, map_location=torch.device(device))
    xmodel.load_state_dict(xload)
    del xload
    #check if want to modularize
    # if feature_extract:
    #     xmodel.eval()
    set_parameter_requires_grad(xmodel, feature_extract)
    n_features = xmodel.config.hidden_size    
    # xmodel = _adapt_model_downstream(xmodel, n_features, num_classes)
    xmodel = CIFAR10Benchmark(xmodel, num_classes)
    return xmodel
        
    
def get_optimizer(model, feature_extract):
    params_to_update = model.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name,param in model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t",name)
    else:
        for name,param in model.named_parameters():
            if param.requires_grad == True:
                print("\t",name)

    if feature_extract:
        lr = 1e-3
        weight_decay =5e-5
        optimizer_ft = torch.optim.Adam(params_to_update, lr=lr, weight_decay=weight_decay)
    #     optimizer_ft = torch.optim.SGD([{'params': params_to_update, 
    #                               'initial_lr':lr}], 
    #                             lr=lr, momentum=0.9)
    else:
        lr=1e-4
        optimizer_ft = torch.optim.Adam(params_to_update, lr=lr)
        
    return optimizer_ft


    
def setup(rank, world_size):    
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'    
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()
    
def DDP_process(rank, world_size, seed, args):#protocol, seed):


    outdir = args.outdir
    model_fpath = args.model_path
    task = args.task
    num_epochs = args.epochs

    torch.cuda.set_device(rank)
    torch.manual_seed(seed)
    
    data = {
        'results': {'train_loss':[]}   
    }
    # we have to create enough room to store the collected objects
    outputs = [None for _ in range(world_size)]
    
     # directory names etc
    #---------------
    
    os.environ['OPENBLAS_NUM_THREADS'] = '20' #@@@@ to help with the num_workers issue
    
    # setup the process groups
    setup(rank, world_size) 
    print('Workers assigned.')
    is_main_proc = is_main_process()
    print('is_main_proc', is_main_proc) #@@@@
    setup_for_distributed(is_main_proc)
    
    number_of_cpu = len(os.sched_getaffinity(0))#multiprocessing.cpu_count() #joblib.cpu_count()
    
    #Instantiate the dataset, criterion
    datasets, num_classes = make_dataset(task)
    feature_extract = True
    model_type = 'res50'
    #-----------------
    # Create the criterion
    criterion = torch.nn.CrossEntropyLoss()
    
    
    # Instantiate the model, optimizer
    #Load the model, adapt it to the downstream task
    xmodel = get_model(model_fpath, num_classes, rank,
                       feature_extract=feature_extract, model_type=model_type) 
    xmodel = xmodel.to(rank)
    xmodel = DDP(xmodel, device_ids=[rank], output_device=rank, 
                   find_unused_parameters=False)
    
    scheduler = None #torch.optim.lr_scheduler.ExponentialLR(
    #     optimizer_ft,gamma=0.9, last_epoch=num_epochs)
    optimizer = get_optimizer(xmodel, feature_extract) 

    # Make the dataloaders and samplers
    sampler_shuffle = True #for the distributed dampler
    batch_size = 128
    pin_memory = True
    num_epochs = num_epochs
    num_workers = 7 #number_of_cpu-1#32
    print('n cpu: ', number_of_cpu, ' n workers: ', num_workers)
    
    samplers_dict = {x: DistributedSampler(datasets[x], num_replicas=world_size, 
                                           rank=rank, shuffle=sampler_shuffle, 
                                           seed=seed)
                     for x in ['train', 'val']}
    dataloaders_dict = {x: torch.utils.data.DataLoader(
        datasets[x], batch_size=batch_size, pin_memory=pin_memory, 
        num_workers=num_workers, shuffle=False, sampler=samplers_dict[x],
        prefetch_factor=int(1.5*batch_size/num_workers))
                        for x in ['train', 'val']}
    
    outfpath = Path(model_fpath).stem+'_downstream_'+task+'.csv'
    if len(outdir)>0:
        outfpath = os.path.join(outdir, outfpath)
    
    train_classifier_ddp(world_size, rank, xmodel, dataloaders_dict, criterion, 
                         optimizer, outfpath, num_epochs=num_epochs, 
                         scheduler=scheduler, verbose=True)
    
    #Where to put nvidia-smi
    print("GPU Memory occupied :-")
    torch.cuda.memory_allocated()
    cleanup()
    
    
#____________________________________________
if __name__ == '__main__':
    
    #---------- Parse the arguments
    parser = argparse.ArgumentParser(description='Evaluate downstream performance for a pretrained model.')

    # Add the arguments

    parser.add_argument('--model_path',
                           type=str,
                           default='random',
                           help='absolute path to the checkpoint file of the pretrained model')

    parser.add_argument('--outdir',
                           type=str,
                           default='',
                           help='')

    parser.add_argument('--epochs',
                           type=int,
                           default=30,
                           help='Number of epochs of training on the downstream task')

    parser.add_argument('--task',
                           type=str,
                           default='toybox',
                           help='')
    #----------


    # Execute the parse_args() method
    args = parser.parse_args()
    
    
   #-------------
    
    n_gpu = torch.cuda.device_count()
    world_size= n_gpu
    seed = np.random.randint(1000)

    mp.spawn(
            DDP_process,
            args=(world_size, seed, args),#prot_arg, seed_arg),
            nprocs=world_size
        )