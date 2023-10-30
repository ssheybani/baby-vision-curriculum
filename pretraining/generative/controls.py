from torch.utils.data import Dataset
import torchvision
import pickle
from homeview import (_get_transform, get_fpathlist,
    get_fold, get_train_val_split, get_group, get_fpathseqlist, 
    ImageSequenceDataset)
import random

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
    

def prepend_jpgroot(gx_relpathseqlist, jpg_root):
    for i in range(len(gx_relpathseqlist)):
        gx_relpathseqlist[i] = [jpg_root + element 
                                for element in gx_relpathseqlist[i]]
    return gx_relpathseqlist


def make_dataset_spatial(subj_dirs, image_size, args):
#                  train_group,
#                  num_frames, jpg_root, fold, condition,
#                 n_trainsamples):
    
    train_group = args.train_group
    if args.num_frames==16:
        control_data_root = '/N/project/baby_vision_curriculum/tmp_data/simple_sequences/seqlen16_fps30/'
    elif args.num_frames==1:
        control_data_root = '/N/project/baby_vision_curriculum/tmp_data/simple_sequences/seqlen1_fps30/'
    else:
        raise ValueError
        
    gx_pkl_fp = control_data_root+train_group+'_samples.pkl'
    
    with open(gx_pkl_fp, 'rb') as file:
        gx_pathseqlist = pickle.load(file)
    
    jpg_root = args.jpg_root #kwargs['jpg_root']
#     image_size = kwargs['image_size']
    fold = args.fold #kwargs['fold']
    condition = args.condition #kwargs['condition']
    n_trainsamples = args.n_trainsamples
    
    transform = _get_transform(image_size)
#     gx_fpathlist = []
#     for i_subj, subjdir in enumerate(tqdm(subj_dirs)):
#         gx_fpathlist += get_fpathlist(jpg_root, subjdir, ds_rate=ds_rate)
    
    max_folds = 3
    gx_pathseqlist = get_fold(gx_pathseqlist, fold, max_folds, args)
    gx_pathseqlist = prepend_jpgroot(gx_pathseqlist, jpg_root)
#     if args.num_frames<16:
#         gx_pathseqlist = adapt_numframes(gx_pathseqlist, args.num_frames) #created for accommodating num_frames=1
    print('Num. samples in the fold:',len(gx_pathseqlist))

    #     if len(gx_fpathlist)>=n_groupframes:
#         gx_fpathlist = gx_fpathlist[:n_groupframes]
#         # 1450000/16 = 90625 => n_trainsamples=81560, n_valsamples= 9060
#         # 1274 iterations of train. 141 iterations of test. 
#         n_trainsamples = None
#         n_valsamples = None
#     else:
    
    # Train-val split
    gx_train_fp, gx_val_fp = get_train_val_split(gx_pathseqlist, val_ratio=0.1)

    if condition=='longshuffle':
        raise NotImplementedError
#         random.shuffle(gx_train_fp)
    
    n_trainsamples = n_trainsamples #int(0.9*n_groupframes/seq_len) #81k
    
    n_maxvalsamples = int(len(gx_val_fp)/1)#seq_len)
    n_valsamples = min(n_maxvalsamples, 10000)  #means don't do bootstraping for val. Use whatever number 0.1*len(gx_fpathlist) gives.
    
    gx_train_fpathseqlist = random.sample(gx_train_fp, n_trainsamples)
    gx_val_fpathseqlist = random.sample(gx_val_fp, n_valsamples)
    
    
    if condition=='shuffle':
        raise NotImplementedError
#         train_dataset = ImageSequenceDataset(gx_train_fpathseqlist, transform=transform, shuffle=True)
#         val_dataset = ImageSequenceDataset(gx_val_fpathseqlist, transform=transform, shuffle=False)
    
    elif condition=='static': #@@@ assumes num_frames=16
        train_dataset = StillVideoDataset(gx_train_fpathseqlist, transform=transform)
        val_dataset = ImageSequenceDataset(gx_val_fpathseqlist, transform=transform, shuffle=False)
#         StillVideoDataset(gx_val_fpathseqlist, transform=transform)

    else:
        train_dataset = ImageSequenceDataset(gx_train_fpathseqlist, transform=transform, shuffle=False)
        val_dataset = ImageSequenceDataset(gx_val_fpathseqlist, transform=transform, shuffle=False)
        
    return {'train':train_dataset,
           'val': val_dataset}