import sys, os

env_root = '/N/project/baby_vision_curriculum/pythonenvs/hfenv/lib/python3.10/site-packages/'
sys.path.insert(0, env_root)

# import numpy as np
import torch, torchvision
from torchvision import transforms as tr
from torch.utils.data import Dataset
import itertools
import random
from pathlib import Path
from PIL import ImageFilter
from tqdm import tqdm


def make_dataset(subj_dirs, image_size, args):
    seq_len = args.num_frames #kwargs['seq_len']
#     n_groupframes=kwargs['n_groupframes']#1450000
    ds_rate = args.ds_rate #kwargs['ds_rate']
    jpg_root = args.jpg_root #kwargs['jpg_root']
#     image_size = kwargs['image_size']
    fold = args.fold #kwargs['fold']
    condition = args.condition #kwargs['condition']
    n_trainsamples = args.n_trainsamples
    
    transform = _get_transform(image_size)
    gx_fpathlist = []
    for i_subj, subjdir in enumerate(tqdm(subj_dirs)):
        gx_fpathlist += get_fpathlist(jpg_root, subjdir, ds_rate=ds_rate)
    
    # added on May15
    max_folds = 3
    gx_fpathlist = get_fold(gx_fpathlist, fold, max_folds, args)
    print('Num. frames in the fold:',len(gx_fpathlist))

    #     if len(gx_fpathlist)>=n_groupframes:
#         gx_fpathlist = gx_fpathlist[:n_groupframes]
#         # 1450000/16 = 90625 => n_trainsamples=81560, n_valsamples= 9060
#         # 1274 iterations of train. 141 iterations of test. 
#         n_trainsamples = None
#         n_valsamples = None
#     else:
    
    # Train-val split
    if hasattr(args, 'keep_val'):
        if args.keep_val=='y':
            val_ratio = 0.1
        else:
            val_ratio = 0
    else:
        val_ratio = 0.01
        
    if val_ratio==0:
        gx_train_fp = gx_fpathlist
        gx_val_fp = []
    else:
        gx_train_fp, gx_val_fp = get_train_val_split(
            gx_fpathlist, val_ratio=val_ratio)

    # if condition=='longshuffle':
    #     random.shuffle(gx_train_fp)
    
    # Make sure we don't bootstrap to increase the number of val samples.
    # We also don't want too many val samples, it makes training longer.
    n_maxvalsamples = int(len(gx_val_fp)/seq_len)
    n_valsamples = min(n_maxvalsamples, 10000)  #means don't do bootstraping for val. Use whatever number 0.1*len(gx_fpathlist) gives.
    
    gx_train_fpathseqlist = get_fpathseqlist(gx_train_fp, seq_len, ds_rate=1, n_samples=n_trainsamples)
    train_dataset = ImageSequenceDataset(gx_train_fpathseqlist, transform=transform, shuffle=False)
    
    if n_valsamples>0:
        gx_val_fpathseqlist = get_fpathseqlist(gx_val_fp, seq_len, ds_rate=1, n_samples=n_valsamples)
        val_dataset = ImageSequenceDataset(gx_val_fpathseqlist, transform=transform, shuffle=False)
    else:
        val_dataset = None
    
    return {'train':train_dataset,
           'val': val_dataset}

    # if condition=='shuffle':
    #     train_dataset = ImageSequenceDataset(gx_train_fpathseqlist, transform=transform, shuffle=True)
    #     val_dataset = ImageSequenceDataset(gx_val_fpathseqlist, transform=transform, shuffle=False)
    
    # elif condition=='static':
    #     train_dataset = StillVideoDataset(gx_train_fpathseqlist, transform=transform)
    #     val_dataset = ImageSequenceDataset(gx_val_fpathseqlist, transform=transform, shuffle=False)
#         StillVideoDataset(gx_val_fpathseqlist, transform=transform)

    

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

def get_fpath2framelist(fpathlist, interval, n_samples=None):
    
    if n_samples is None:
        n_samples = len(fpathlist)-interval-1
        sample_stride = 1
    else:
        assert len(fpathlist)>=n_samples
        sample_stride = int((len(fpathlist)-interval-1)/n_samples)
    
    fpath2framelist = [[fpathlist[i], fpathlist[i+interval]]
                    for i in range(0, n_samples*sample_stride, sample_stride)]
    return fpath2framelist

        
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


def get_fold(gx_fpathlist, fold, max_folds, args):
#     fold_size = int(len(gx_fpathlist)/max_folds)
    segment_size = int(30*60*30/args.ds_rate)
    
    fold_segments = []

    for i_st in range(0, len(gx_fpathlist), segment_size):
        if (i_st // segment_size) % max_folds == fold:
            fold_segments.append(gx_fpathlist[i_st:i_st + segment_size])
            
    fold_segments = list(itertools.chain.from_iterable(fold_segments))
    return fold_segments


def get_group(train_group):
    # n_groupframes = 1450000 # minimum number of frames across age groups
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
    group = group_dict.get(train_group)
    return group

def get_color_distortion(s=1.0):
        # s is the strength of color distortion.
        color_jitter = tr.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
        rnd_color_jitter = tr.RandomApply([color_jitter], p=0.8)
        rnd_gray = tr.RandomGrayscale(p=0.2)
        color_distort = tr.Compose([
            rnd_color_jitter,
            rnd_gray])
        return color_distort

class GaussianBlur(object):
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        if torch.bernoulli(torch.tensor(self.prob)) == 0:
            return img

        radius = self.radius_min + torch.rand(1) * (self.radius_max - self.radius_min)
        return img.filter(ImageFilter.GaussianBlur(radius=radius))

def _get_transform(image_size):
#     for vjepa_v0

    mean = [0.5, 0.5, 0.5]#np.mean(mean_all, axis=0) #mean_all[chosen_subj] 
    std = [0.25, 0.25, 0.25] #std_all[chosen_subj] 
#     [0.485, 0.456, 0.406]  # IMAGENET_DEFAULT_MEAN
#     [0.229, 0.224, 0.225]  # IMAGENET_DEFAULT_STD
            
#     Alternative: 
    transform_list = [tr.Resize(image_size),
                      tr.CenterCrop(image_size),
                    tr.ConvertImageDtype(torch.float32),
                    tr.Normalize(mean,std)]# crop_scale=(0.3, 1.0),
    return tr.Compose(transform_list)


# class ImageSequenceDataset(Dataset):
import PIL
class ImageDataset(Dataset):
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
        fp = self.image_paths[idx][0]
        images = self.transform(PIL.Image.open(fp))
#         images = self.transform(torchvision.io.read_image(fp))
        return images

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
#         fp = self.image_paths[idx][0]
#         images = self.transform(PIL.Image.open(fp))
#         images = self.transform(torchvision.io.read_image(fp))
        images = torch.cat([
            self.transform(torchvision.io.read_image(fp)).unsqueeze(0)
                     for fp in self.image_paths[idx]]) #with tochvision transform

        if self.shuffle:
            size = images.size(0)
            perm = torch.randperm(size)
            images = images[perm]
            
        return images

class TwoFrameDataset(Dataset):
    """
    To use for video models. 
    """
    def __init__(self, image_paths, transform, interval=None, seq_size=None, shuffle=False):
        self.image_paths = image_paths
        self.transform = transform
        self.shuffle = shuffle #unused in TwoFrameDataset
        self.seq_size = seq_size #unused in TwoFrameDataset
        self.interval = interval #unused in TwoFrameDataset

    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        # Load the sequence of images
#         fp = self.image_paths[idx][0]
#         images = self.transform(PIL.Image.open(fp))
#         images = self.transform(torchvision.io.read_image(fp))
        twoframes = [
            self.transform(PIL.Image.open(fp)).unsqueeze(0)
                     for fp in self.image_paths[idx]]
        images = torch.cat(twoframes)
            
        return images
    
    
class TwoSeqDataset(Dataset):
    """
    To use for video models. 
    """
    def __init__(self, image_paths, transform, interval, seq_size, shuffle=False):
        self.image_paths = image_paths
        self.transform = transform
        self.shuffle = shuffle
        self.seq_size = seq_size #equals tubelet_size
        self.interval = interval

    def __len__(self):
        return len(self.image_paths)-self.interval-self.seq_size
    
    def safe_idx(self, idx, interval):
        new_idx = idx-interval
        if new_idx>self.__len__():
            return idx
        else:
            return new_idx
        
    def __getitem__(self, idx):
        # Load the sequence of images
#         fp = self.image_paths[idx][0]
#         images = self.transform(PIL.Image.open(fp))
#         images = self.transform(torchvision.io.read_image(fp))
        
        seq1 = torch.cat([
            self.transform(torchvision.io.read_image(fp)).unsqueeze(0)
                     for fp in self.image_paths[idx:idx+self.seq_size]])
        
        idx2 = self.safe_idx(idx, self.interval)
        seq2 = torch.cat([
            self.transform(torchvision.io.read_image(fp)).unsqueeze(0)
                     for fp in self.image_paths[idx2:idx2+self.seq_size]])

        images = torch.cat([seq1,seq2])
        
        if self.shuffle:
            size = images.size(0)
            perm = torch.randperm(size)
            images = images[perm]
            
        return images
    
    
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

