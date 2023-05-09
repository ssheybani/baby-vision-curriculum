# Built on top of encode_images_ddp_v4

import sys, os

# env_root = '/N/project/baby_vision_curriculum/pythonenvs/hfenv/lib/python3.10/site-packages/'
# sys.path.insert(0, env_root)

if 'BigRed200' in os.getcwd().split('/'):
    sys.path.insert(0,'/geode2/home/u080/sheybani/BigRed200/spenv/lib/python3.10/site-packages')
#     print('Running on BigRed200')

# import torch, torchvision
import numpy as np
from tqdm import tqdm
from pathlib import Path
# from torchvision import transforms as tr
# import pandas as pd
import argparse
# from itertools import chain

# import torch.distributed as dist
# from torch.utils.data.distributed import DistributedSampler
# from torch.nn.parallel import DistributedDataParallel as DDP
# import torch.multiprocessing as mp
import skimage, joblib
import warnings

from skimage.transform import rescale
from skimage.util import img_as_ubyte
from skimage.color import rgb2gray
from skimage.io import imread, imsave

import pickle #dill
# import timm




# def get_transform(image_size, crop_size):

#     # transform = tr.Compose([
#     #     tr.Resize(image_size),
#     #     tr.CenterCrop(image_size),
#     #     tr.ConvertImageDtype(torch.float32),
#     #     tr.Normalize(mean,std)
#     # ])
#     def xtransform(img): 
        
#         if len(img.shape)==2:
#             img = img[:,:,np.newaxis]#np.expand_dim(img,2)
#         h,w, c = img.shape
        
#         img_res = rgb2gray(img)
#         scale_f = image_size/min(h,w)
#         img_res = rescale(img_res, scale_f, anti_aliasing=True)
#         img_res = img_as_ubyte(img_res)
#         img_res = center_crop(img_res, crop_size)
        
#         return img_res
        
#     return xtransform

class ImageTransform:
    def __init__(self, image_size, crop_size):
        self.image_size = image_size
        self.crop_size = crop_size
    
    def __call__(self, img):
        if len(img.shape) == 2:
            img = img[:, :, np.newaxis]
        h, w, c = img.shape

        img_res = rgb2gray(img)
        scale_f = self.image_size / min(h, w)
        img_res = rescale(img_res, scale_f, anti_aliasing=True)
        img_res = img_as_ubyte(img_res)
        img_res = self.center_crop(img_res, self.crop_size)

        return img_res

    def center_crop(self, img, crop_size):
        h, w = img.shape
        h_start = (h - crop_size) // 2
        w_start = (w - crop_size) // 2
        h_end = h_start + crop_size
        w_end = w_start + crop_size
        return img[h_start:h_end, w_start:w_end]


def get_fpathlist(vid_root, subjdir, ds_rate=1):
    # read the image file names into fpathlist
    # subjdir = ['008MS'] #@@@

    # vid_root = r"/N/project/infant_image_statistics/03_JPG_AllMasters/"
    
    fpathlist = sorted(list(Path(os.path.join(vid_root, subjdir)).iterdir()), 
                       key=lambda x: x.name)
    fpathlist = [str(fpath) for fpath in fpathlist if fpath.suffix=='.jpg']
    fpathlist = fpathlist[::ds_rate]
    return fpathlist
    
def _proc_image(fpath, transform, outdir):

    # load
    img = imread(fpath)
    if img is None:
        warnings.warn('Image file at '+fpath+' could not be read!')
        return
    # transform (Reize)
    if transform is not None:
        img = transform(img)
    fname = Path(fpath).name#fpath.split('\')[-1]
#     print('fname:',fname)
    save_path = os.path.join(outdir,fname)
#     print(save_path)
    imsave(save_path, img)
    
    # return None
    
def is_picklable(obj):
    try:
        pickle.dumps(obj)
#         dill.dumps(obj)#
        return True
    except:
        return False



def proc_subj(world_size, subjdir, args):
    
    image_size, crop_size = args.image_size, args.crop_size
#     def xtransform(img): 
#         if len(img.shape)==2:
#             img = img[:,:,np.newaxis]#np.expand_dim(img,2)
#         h,w, c = img.shape
        
#         img_res = rgb2gray(img)
#         scale_f = image_size/min(h,w)
#         img_res = rescale(img_res, scale_f, anti_aliasing=True)
#         img_res = img_as_ubyte(img_res)
#         img_res = center_crop(img_res, crop_size)
        
#         return img_res
    
    transform = ImageTransform(args.image_size, args.crop_size)#xtransform
    fpathlist = get_fpathlist(args.jpg_root, subjdir, ds_rate=args.ds_rate)
    # fnamelist = [item.split('/')[-1] for item in fpathlist]
    
    outdir = os.path.join(args.save_dir,subjdir)
    
    Path(outdir).mkdir(parents=True, exist_ok=True)
    
    print('is_picklable(transform):', is_picklable(transform))
    print('is_picklable(outdir):', is_picklable(transform))
    
    for item in fpathlist:
        if is_picklable(item)==False:
            print(item, 'not picklable')
    print('no unpicklable item in fpathlist')
    
    joblib.Parallel(n_jobs=world_size, verbose=1)(
        joblib.delayed(_proc_image)(item, transform, outdir) 
        for item in tqdm(fpathlist))
    
    # allembeddings = np.asarray(allembeddings)
    # print('result shape:',allembeddings.shape)

    # Save the file
    
    print('Finished processing ', subjdir, ' saved in ', outdir)
            
    return
  

    
if __name__ == '__main__':
    
    #---------- Parse the arguments
    parser = argparse.ArgumentParser(description='Embedd Images')

    # Add the arguments
    parser.add_argument('-save_dir',
                           type=str,
                           help='directory to save the results')
    
    parser.add_argument('--jpg_root',
                       type=str,
                       default='/N/project/infant_image_statistics/preproc_saber/JPG_10fps/',
                       help='The root of all image directories')
    
    parser.add_argument('--subj_dirs',
                       type=str,
                    default='all',
                       help='name of the subject directories, split by +')
    
    parser.add_argument('--image_size',
                           type=int,
                           default=256,
                           help='the image is resize+cropped to this number before embedding')
    
    parser.add_argument('--crop_size',
                           type=int,
                           default=32,
                           help='a crop from the center of the image is used.')
    
    parser.add_argument('--ds_rate',
                           type=int,
                           default=1,
                           help='the embeddings are computed for every ith image')
    
    args = parser.parse_args()

    #--------------------------
    
    
    # Figure out hardware
    number_of_cpu = len(os.sched_getaffinity(0))#multiprocessing.cpu_count() #joblib.cpu_count()
    world_size= min(80, number_of_cpu)
    
    if args.subj_dirs!='all':
        subjname_list = args.subj_dirs.split('+')
    else:
        subjname_list = list(os.listdir(args.jpg_root))
    subj_dirs = sorted([item for item in subjname_list
                        if os.path.isdir(args.jpg_root+item)])
    
    print('subj_dirs: ', subj_dirs)
    for i_subj, subjdir in enumerate(tqdm(subj_dirs)):
        print('Processing subject ',i_subj,':',subjdir)
        proc_subj(world_size, subjdir, args)