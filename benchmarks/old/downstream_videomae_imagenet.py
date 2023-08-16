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
from PIL import Image
import transformers
from sklearn import preprocessing
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from ddputils import is_main_process, save_on_master, setup_for_distributed

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

def _get_transform(image_size):

    mean = [0.5, 0.5, 0.5]#np.mean(mean_all, axis=0) #mean_all[chosen_subj] 
    std = [0.25, 0.25, 0.25] #std_all[chosen_subj] 
    
#     [0.485, 0.456, 0.406]  # IMAGENET_DEFAULT_MEAN
#     [0.229, 0.224, 0.225]  # IMAGENET_DEFAULT_STD

    augs = [tr.Resize(image_size), tr.CenterCrop(image_size), 
            tr.ConvertImageDtype(torch.float32), 
             tr.Normalize(mean,std)]
    return tr.Compose(augs)

def transform_image(image):
#     Used for standard single image datasets such as torchvision.CIFAR10, torchvision.ImageNet
#     if image.shape[0]!=3:
    image_size=224
    num_frames=16
    transform = _get_transform(image_size)
    
    if image.shape[0] == 1:
        image = image.repeat(3, 1, 1)
    
    image_tr = transform(image)

    return image_tr.unsqueeze(0).repeat(num_frames,1,1,1)

# def get_fnames_labels(csv_file):
#     data = pd.read_csv(csv_file).sort_values('target')
#     print(data['filename'])
#     le = preprocessing.LabelEncoder()
#     fnames = data['filename']
#     labels = le.fit_transform(data['target'])
#     synsets = data['target']
#     return fnames, labels, synsets
def get_fnames_synsets(csv_file):
    data = pd.read_csv(csv_file).sort_values('target')
#     le = preprocessing.LabelEncoder()
    fnames = data['filename']
#     labels = le.fit_transform(data['target'])
    synsets = data['target']
    return fnames.tolist(), synsets.tolist()

def get_synset_encoder(csv_fp):
    data = pd.read_csv(csv_fp).sort_values('target')
    synsets = data['target']
    le = preprocessing.LabelEncoder()
    le = le.fit(synsets)
    return le
class TrainDataset(Dataset):
    def __init__(self, csv_file, root_dir, label_encoder, transform=None):
        self.fnames, self.synsets = get_fnames_synsets(csv_file)
        self.labels = label_encoder.transform(self.synsets)
        print(type(self.labels))
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        img_fp = f"{self.root_dir}{self.synsets[idx]}/{self.fnames[idx]}"
#         img_name = f"{self.root_dir}/{self.data.iloc[idx, 1]}/{self.data.iloc[idx, 0]}"
        image = torchvision.io.read_image(img_fp)
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
#         print(image.shape)

        return image, label

class ValDataset(Dataset):
    def __init__(self, csv_file, root_dir, label_encoder, transform=None):
        self.fnames, self.synsets = get_fnames_synsets(csv_file)
        self.labels = label_encoder.transform(self.synsets)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        img_fp = f"{self.root_dir}{self.fnames[idx]}"
#         img_name = f"{self.root_dir}/{self.data.iloc[idx, 1]}/{self.data.iloc[idx, 0]}"
        image = torchvision.io.read_image(img_fp)
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
#         print(image.shape)

        return image, label
def make_imagenet_dataset(args):
    imagenet_root='/N/project/baby_vision_curriculum/benchmarks/mainstream/imagenet/ILSVRC/Data/CLS-LOC/'
#     val_path = '/N/project/baby_vision_curriculum/benchmarks/mainstream/imagenet/LOC_val_solution.csv'
    val_path = '/N/project/baby_vision_curriculum/github/baby-vision-curriculum/benchmarks/imagenet/val_targets.csv'
    train_path = '/N/project/baby_vision_curriculum/github/baby-vision-curriculum/benchmarks/imagenet/train_targets.csv'
    transform = transform_image
    label_encoder = get_synset_encoder(train_path)
    train_dataset = TrainDataset(train_path, f"{imagenet_root}train/",
                                                            label_encoder, transform=transform)
    val_dataset =  ValDataset(val_path, f"{imagenet_root}val/", 
                              label_encoder, transform=transform)
    image_datasets = {'train': train_dataset , 'val':val_dataset}
    num_classes = 1000 
        
    return image_datasets, num_classes

def get_config(image_size, args, num_labels=2):
    arch_kw = args.architecture
    if arch_kw=='small2':
        hidden_size = 768
        intermediate_size = 4*768
        num_attention_heads = 6
        num_hidden_layers = 6
        
        config = transformers.VideoMAEConfig(image_size=image_size, patch_size=16, num_channels=3,
                                             num_frames=16, tubelet_size=2, 
                                             hidden_size=hidden_size, num_hidden_layers=num_hidden_layers, num_attention_heads=num_attention_heads,
                                             intermediate_size=intermediate_size, num_labels=num_labels)
    
    elif arch_kw=='base': #default
        config = transformers.VideoMAEConfig(image_size=image_size, patch_size=16, num_channels=3,
                                             num_frames=16, tubelet_size=2, 
                                             hidden_size=768, num_hidden_layers=12, num_attention_heads=12,
                                             intermediate_size=3072, num_labels=num_labels)
    elif arch_kw=='small1':
        hidden_size = 384
        intermediate_size = 4*384
        num_attention_heads = 6
        num_hidden_layers = 12
        
        config = transformers.VideoMAEConfig(image_size=image_size, patch_size=16, num_channels=3,
                                             num_frames=16, tubelet_size=2, 
                                             hidden_size=hidden_size, num_hidden_layers=num_hidden_layers, num_attention_heads=num_attention_heads,
                                             intermediate_size=intermediate_size, num_labels=num_labels)
        
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
    
    if feature_extracting:
        set_parameter_requires_grad(model_target, feature_extracting)
    
    return model_target

        
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
def get_optimizer(model, feature_extract, args):
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

#     if feature_extract:
    lr = args.lr#1e-3
    weight_decay =args.wd#5e-5
    optimizer_ft = torch.optim.Adam(params_to_update, lr=lr, weight_decay=weight_decay)
    #     optimizer_ft = torch.optim.SGD([{'params': params_to_update, 
    #                               'initial_lr':lr}], 
    #                             lr=lr, momentum=0.9)
#     else:
#         lr=1e-4
#         optimizer_ft = torch.optim.Adam(params_to_update, lr=lr)
        
    return optimizer_ft

def make_dataset(args):
    task = args.task
    # if task=='ucf101':
    #     return make_ucf101dataset(args)
#     seq_len = kwargs['seq_len']
#     image_size = kwargs['image_size']
    # elif task=='cifar10':
    #     return make_cifar10dataset(args)
    if task=='imagenet':
        return make_imagenet_dataset(args)
    else:
        raise NotImplementedError()

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
    datasets, num_classes = make_dataset(args)
    feature_extract = True
#     model_type = 'res50'
    #-----------------
    # Create the criterion
#     criterion = torch.nn.CrossEntropyLoss()
    
    
    # Instantiate the model, optimizer
    #Load the model, adapt it to the downstream task
    image_size = 224
    feature_extract = (args.finetune=='n')
    xmodel = get_model(image_size, num_classes, feature_extract, args)
    
    xmodel = xmodel.to(rank)
    xmodel = DDP(xmodel, device_ids=[rank], output_device=rank, 
                   find_unused_parameters=False)
    
    # scheduler = None #torch.optim.lr_scheduler.ExponentialLR(
    #     optimizer_ft,gamma=0.9, last_epoch=num_epochs)
    optimizer = get_optimizer(xmodel, feature_extract, args) 

    # Make the dataloaders and samplers
    sampler_shuffle = True #for the distributed dampler
    num_epochs = args.n_epoch
    batch_size = args.batch_size# 128
    pin_memory = False #True
    num_workers = args.num_workers #number_of_cpu-1#32
    
    # if args.task=='ucf101':
    #     collate_fn = ucf_collate
    # else:
    collate_fn = None
    print('n cpu: ', number_of_cpu, ' n workers: ', num_workers)
    
    samplers_dict = {x: DistributedSampler(datasets[x], num_replicas=world_size, 
                                           rank=rank, shuffle=sampler_shuffle, 
                                           seed=seed)
                     for x in ['train', 'val']}
    print("Collate function", collate_fn)
    dataloaders = {x: torch.utils.data.DataLoader(
        datasets[x], batch_size=batch_size, pin_memory=pin_memory, collate_fn=collate_fn,
        num_workers=num_workers, shuffle=False, sampler=samplers_dict[x], drop_last=True)
                        for x in ['train', 'val']}
    print('len dset, len dloader: ', len(datasets['train']), len(dataloaders['train']))
#         print(dataset.__getitem__(22).shape)
    print('dataloaders created') #@@@
    
    verbose = (verbose and is_main_process())
#     identifying which checkpoint was benchmarked
# <model vs scores>_<prot>_seed_<seed>_<other id>
    
#     train_classifier_ddp(world_size, rank, xmodel, dataloaders_dict, criterion, 
#                          optimizer, outfpath, num_epochs=num_epochs, 
#                          scheduler=scheduler, verbose=True)
    train_acc_history = []
    val_acc_history = []
    
    # best_model_wts = deepcopy(model.state_dict())
    best_acc = 0.0

    for i_ep in range(num_epochs):
        if verbose:
            print('Epoch {}/{}'.format(i_ep, num_epochs - 1))
            print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
#             print("Here")
            dataloaders[phase].sampler.set_epoch(i_ep)
#             print("I'm here")
            if phase == 'train':
                xmodel.train()  # Set model to training mode
            else:
                xmodel.eval()   # Set model to evaluate mode

            running_loss = torch.tensor([0.0], device=rank)
            running_corrects = torch.tensor([0.0], device=rank)

            i_iter, print_period=0, 100  
#             i_break, print_period = 20,5 #@@@ debug
            # Iterate over data.
            print("phase", phase)
    #         dataiter = iter(dataloaders[phase])
    #         images, labels = next(dataiter)
    #         print(images.shape)
            for batch in tqdm(dataloaders[phase]):
                # zero the parameter gradients
                optimizer.zero_grad()

    #                 loss, logits = get_loss(task, batch, phase, rank, args)

                # implement get_loss for different datasets and for videomaeclassifier
                inputs, labels = batch #get_inp_label(args.task, batch)
    #             print("Here", inputs.shape)
                inputs = inputs.to(rank)
                labels = labels.to(rank)
                outputs = xmodel(pixel_values=inputs, labels=labels)

                logits = outputs.logits
                loss = outputs.loss

                _, preds = torch.max(logits, 1)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
    #                 print(rank, 'labels shape, device: ', labels.shape, labels.data.device)
    #                 print(rank, 'preds shape, device: ', preds.shape, preds.device)
                running_corrects += torch.sum(preds == labels.data)

                i_iter+=1
                if (i_iter%print_period)==0:
                    print('loss:',loss.item())
                    memory_allocated = torch.cuda.memory_allocated() / 1024**2
                    print(f'GPU memory allocated: {memory_allocated:.2f} MB')

#                 if i_iter==i_break:
#                     break #@@@@ debug
        
            # reduce running_loss, running_corrects after exhausting the dataloader
            dist.barrier()
            # dist.reduce(tensor,dst,dis.ReduceOp,group)
            dist.all_reduce(running_loss, op=dist.ReduceOp.SUM, async_op=False)
            dist.all_reduce(running_corrects, op=dist.ReduceOp.SUM, async_op=False)
            
            if is_main_process():
                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
                
                if verbose:
                    print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss.item(), epoch_acc.item()))

                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    # best_model_wts = deepcopy(model.state_dict())
                if phase == 'val':
                    val_acc_history.append(epoch_acc.cpu().item())
                else:
                    train_acc_history.append(epoch_acc.cpu().item())

        print()
    if verbose:
        print('Training complete')
        print('Best val Acc: {:4f}'.format(best_acc.item()))

    results_df = pd.DataFrame({'epoch': np.arange(len(train_acc_history)),
                               'train_acc':train_acc_history,
                               'val_acc': val_acc_history})

    if is_main_process():
        savedir = args.savedir
        Path(savedir).mkdir(parents=True, exist_ok=True)
#         <model vs scores>_<prot>_seed_<seed>_other_<other>_<other id>
        result_fname = '_'.join(['scores', args.prot_name, 
                                'seed', str(seed),  
                                args.other_id])+'_.csv'
        results_fpath = os.path.join(savedir, result_fname)
        results_df.to_csv(results_fpath, sep=',', float_format='%.3f')
        # np.savetxt(outfpath, np.asarray(acc_history).T, delimiter=',', fmt='%.3f')
        if args.save_model!='n':
            model_fname = '_'.join(['model', args.prot_name, 
                                   'seed', str(seed),
                                   args.other_id])+'.pt'
            MODELPATH = os.path.join(savedir, model_fname)
            SCRIPT = ''#args.script
            torch.save({
                'init_checkpoint_path':args.init_checkpoint_path,
                'model_state_dict': xmodel.module.state_dict(),
                'script': SCRIPT,
                # 'train_sets': TRAIN_SETS
                }, MODELPATH)
#         'optimizer_state_dict': optimizer.state_dict(),
            print('model saved at ',MODELPATH)
        

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
    parser.add_argument('--lr',
                           type=float,
                           default=1e-3,
                           help='')
    parser.add_argument('--wd',
                           type=float,
                           default=5e-5,
                           help='')
    parser.add_argument('--batch_size',
                           type=int,
                           default=16,
                           help='')
    parser.add_argument('--num_workers',
                           type=int,
                        default=6,
                           help='number of cpu for each gpu')
    
    parser.add_argument('--architecture',
                           type=str,
                           default='',
                           help='see get_config')    
        
    parser.add_argument('--n_epoch',
                           type=int,
                           default=30,
                           help='')
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