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

# from transformers import VideoMAEConfig, VideoMAEModel
# from torch.utils.data import Dataset
# import av

# from time import time
# from copy import deepcopy
import cv2
from itertools import chain

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



class ToyboxDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform, frame_rate=10, sample_len=16):
        self.root_dir = root_dir
        self.frame_rate = frame_rate
        self.sample_len = sample_len
        self.transform = transform
        self.samples = []
        for supercategory in os.listdir(self.root_dir):
            for obj in os.listdir(os.path.join(self.root_dir, supercategory)):
#                 for obj in os.listdir(os.path.join(self.root_dir, supercategory, category)):
                object_dir = os.path.join(self.root_dir, supercategory, obj)
                for view in os.listdir(object_dir):
                    view_path = os.path.join(object_dir, view)
                    self.samples.append(view_path)
#                         self.samples.append((view_path, supercategory, category, object))

    def __len__(self):
        return len(self.samples)

    def get_all_frames(self, cap):
        desired_frames = self.sample_len
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
#                 print('end of the video, i_frame, len fames', frame_count, len(frames))
                # End of video
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)                
            if len(frames) == desired_frames:
                break
        tmp_nframes = len(frames)
        if tmp_nframes < desired_frames:
            last_frame = frames[-1]
            for i in range(desired_frames - tmp_nframes):
                frames.append(last_frame)
        
        assert len(frames)==desired_frames
        return frames
    
    def wrap_frames(self, frames):
        frames = torch.as_tensor(np.asarray(frames))
        if len(frames.shape)!=4: #torch.Size([16, 12xx, 19xx, 3])
            return None
        return self.transform(frames)
            
    def __getitem__(self, index):
#         print('---------------')
        vid_path = self.samples[index]
        vid_fname = Path(vid_path).name
        frames = []
        cap = cv2.VideoCapture(vid_path)
        if cap is None or not cap.isOpened():
            warnings.warn('unable to open video source: '+vid_path)
            return None, None

        fps = cap.get(cv2.CAP_PROP_FPS)
        ds_rate = round(fps/self.frame_rate)
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#         print('num_frames:',num_frames)
#         print('ds_rate:',ds_rate)
#         print('num_frames:',num_frames)
        
        sample_scope = self.sample_len*ds_rate
        if num_frames<sample_scope:
#             print('Not enough frames in the video',vid_path)
            frames = self.get_all_frames(cap)
                        #apply transform
            frames_transformed = self.wrap_frames(frames)
            if frames_transformed is None:
                print(vid_path, 'gave None')
                return None, None
            return frames_transformed, vid_fname
            
        
        # duration = num_frames / fps
        start_frame = int(num_frames * 1 / 5)  # Starting frame at 2/3 of video duration
        if (num_frames-start_frame)<sample_scope:
            start_frame = num_frames-sample_scope
        
#         print('start_frame',start_frame)
#         end_frame = start_frame+sample_scope#int(start_frame + fps * 1.6)  # Ending frame after 1.6 seconds
        desired_frames = self.sample_len
        frame_count = 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
#                 print('end of the video, i_frame, len fames', frame_count, len(frames))
                # End of video
                break
            
            if frame_count % ds_rate==0:
#                 if (frame_count > start_frame) & \
#                 (frame_count < end_frame):
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)                
            if len(frames) == desired_frames:
                break
            frame_count += 1
            
        cap.release()
        frames_transformed = self.wrap_frames(frames)
        
        if frames_transformed is None:
            print(vid_path, 'gave None')
            return None, None
        else:
            return frames_transformed, vid_fname
            
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
                                             num_frames=args.num_frames, tubelet_size=2, 
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
# def adapt_videomae(source_model, target_model):
#     # load the embeddings
#     target_model.embeddings.load_state_dict(
#         source_model.videomae.embeddings.state_dict())
# #     load the encoder
#     target_model.encoder.load_state_dict(
#         source_model.videomae.encoder.state_dict())
#     return target_model

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
#         for param in model.classifier.parameters():
#             param.requires_grad = True
            
def get_model(image_size, num_labels, feature_extracting, args):
    config_source = get_config(image_size, args)
    model_source = transformers.VideoMAEForPreTraining(config_source)
    
    if args.init_checkpoint_path!='na':
        print('args.init_checkpoint_path:',args.init_checkpoint_path)
        # initialize the model using the checkpoint
        model_source = init_model_from_checkpoint(model_source, args.init_checkpoint_path)
  
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
    result_fname = '_'.join(['embeddings', args.prot_name, 
                            'seed', str(args.seed),  
                            args.other_id])+'.csv'
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
    setup_for_distributed(is_main_proc)
    
    number_of_cpu = len(os.sched_getaffinity(0))#multiprocessing.cpu_count() #joblib.cpu_count()
    
    #Instantiate the dataset, criterion
    toybox_root = args.vid_root
#     '/N/project/baby_vision_curriculum/benchmarks/toybox/vids/toybox/'
    transform = transform_vid
    frame_rate=args.frame_rate
    sample_len=args.num_frames
    dataset = ToyboxDataset(toybox_root, transform, 
                          frame_rate=frame_rate, sample_len=sample_len)
#     feature_extract = True
#     model_type = 'res50'
    #-----------------
    # Create the criterion
#     criterion = torch.nn.CrossEntropyLoss()
    
    
    # Instantiate the model, optimizer
    #Load the model, adapt it to the downstream task
    image_size = 224
#     feature_extract = (args.finetune=='n')
    num_classes=0
    feature_extract=True
    xmodel = get_model(image_size, num_classes, feature_extract, args)
    
    xmodel = xmodel.to(rank)
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
    
        save_results(allfnames, allembeddings, args)
        

    cleanup()
    
    
#____________________________________________
if __name__ == '__main__':
    
    #---------- Parse the arguments
    parser = argparse.ArgumentParser(description='Evaluate downstream performance for a pretrained model.')

    # Add the arguments

    parser.add_argument('-vid_root',
                           type=str,
                           help='absolute path to the toybox dataset')
    
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