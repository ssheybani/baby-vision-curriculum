import sys, os

if 'BigRed200' in os.getcwd().split('/'):
    print('Running on BigRed200')
    sys.path.insert(0,'/geode2/home/u080/sheybani/BigRed200/spenv/lib/python3.10/site-packages')

import torch, torchvision
import numpy as np
from tqdm import tqdm
import pickle
# from sklearn import decomposition
from copy import deepcopy
import joblib

import torch.nn as nn

from pathlib import Path

import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import argparse

class ImageSequenceDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load the sequence of images
        images = []
        for path in self.image_paths[idx]:
            image = Image.open(path)
            if self.transform:
                image = self.transform(image).squeeze()
            images.append(image)
             # Convert the sequence of images to a tensor
        images = torch.stack(images)

        return images

def get_transform(image_size, crop_size):

    # transform = tr.Compose([
    #     tr.Resize(image_size),
    #     tr.CenterCrop(image_size),
    #     tr.ConvertImageDtype(torch.float32),
    #     tr.Normalize(mean,std)
    # ])
    transform = transforms.Compose([
        transforms.Resize(image_size),   # Rescale the image to 256 pixels on the shorter side
        transforms.CenterCrop(crop_size),   # Crop the center 224x224 pixels
        transforms.Grayscale(num_output_channels=1),   # Convert the image to grayscale
        transforms.ToTensor(),   # Convert the image to a PyTorch tensor
        transforms.Normalize(0.5,0.5), #@@@ newly added after recAE
        ])
    return transform

class RecAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, seq_len):
        super(RecAutoencoder, self).__init__()
        self.hidden_size = hidden_size
        self.encoder = nn.RNN(input_size, hidden_size, batch_first=True)
        self.decoder = nn.Linear(hidden_size, input_size)
        
        init_r = torch.eye(hidden_size)
        self.encoder.weight_hh_l0.data += 0.1*init_r
        self.encoder.weight_ih_l0.data = 0.5*self.encoder.weight_ih_l0.data
#         nn.init.eye_(self.encoder.weight_hh_l0.data)

    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        # Feed input through the encoder RNN
        out, _ = self.encoder(x, h0)
        # apply the linear layer on each of the steps of the RNN output.
        B,S,H = out.shape
        out = out.reshape(B*S,H)
        out = self.decoder(out)
        return out.view(B,S,input_size)

def perturb_model(model, pr, p_sigma):
    # pr: perturb rate. comparable with learning rate and decay rate.
#                 w_new = w_old*(1-pr) + pr*N(0,1)
    for (name, param) in model.named_parameters():
        if 'weight' in name:   # just weights
            param.data = (1-pr) * param.data + \
            pr* torch.normal(0., p_sigma, size=param.shape)
    return

def get_fpathlist(vid_root, subjdir, ds_rate=1):
    # read the image file names into fpathlist
    # subjdir = ['008MS'] #@@@

    # vid_root = r"/N/project/infant_image_statistics/03_JPG_AllMasters/"
    
    fpathlist = sorted(list(Path(os.path.join(vid_root, subjdir)).iterdir()), 
                       key=lambda x: x.name)
    fpathlist = [str(fpath) for fpath in fpathlist if fpath.suffix=='.jpg']
    fpathlist = fpathlist[::ds_rate]
    return fpathlist

def get_loss(objective, model, images):
    rloss, ploss, dloss = 0,0,0
    
    if 'r' in objective:
        # reconstruction loss
        B, seq_len, H,W = images.shape #512, 2, 8, 8
        images = images.view(B, seq_len, -1)
        reconstructions = model(images)
        rloss = nn.MSELoss()(reconstructions, images)
    
    if 'p' in objective:
        # predictive
        pass
#         B, seq_len, H,W = images.shape #512, 2, 8, 8
#         target = images[:,-1,:,:]
#         inp = images[:,:-1,:,:]
#         inp = inp.view(B, seq_len-1, -1)
#         target = target.view(B,1,-1)
        
    if 'd' in objective:
        pass
    
    total_loss = rloss + ploss + dloss
    return total_loss

if __name__ == '__main__':
    
    #---------- Parse the arguments
    parser = argparse.ArgumentParser(description='Compress image patches using a shallow autoencoder')

    # Add the arguments
    parser.add_argument('-save_path',
                           type=str,
                           help='directory to save the results')
    
    parser.add_argument('-jpg_root',
                       type=str,
                       default='/N/project/infant_image_statistics/preproc_saber/JPG_10fps/',
                       help='The root of all image directories')
    
    parser.add_argument('-subj_dirs',
                       type=str,
                    default='all',
                       help='name of the subject directories, split by +')
    
    parser.add_argument('-objective',
                       type=str,
                    default='r',
                       help='r:reconstruct p:predict d:disentangle')
    
    parser.add_argument('--image_size',
                       type=int,
                    default=64,
                       help='name of the subject directories, split by +')
    
    parser.add_argument('--crop_size',
                       type=int,
                    default=8,
                       help='size of the input image to the model')
    
    parser.add_argument('--seq_len',
                       type=int,
                    default=2,
                       help='')
    
    parser.add_argument('--hidden_size',
                       type=int,
                    default=16,
                       help='dimensionality of the compressed')
    
    parser.add_argument('--num_epochs',
                       type=int,
                    default=10,
                       help='')
    
    args = parser.parse_args()

    


    #--------------------------
    
    
    # Figure out hardware
    number_of_cpu = len(os.sched_getaffinity(0))#multiprocessing.cpu_count() #joblib.cpu_count()
    world_size= min(50, number_of_cpu)
    
    ds_rate = 1
    image_size = args.image_size
    crop_size = args.crop_size
    
    if args.subj_dirs!='all':
        subjname_list = args.subj_dirs.split('+')
    else:
        subjname_list = list(os.listdir(args.jpg_root))
    subj_dirs = sorted([item for item in subjname_list
                        if os.path.isdir(args.jpg_root+item)])
    
    print('subj_dirs: ',subj_dirs)
    
    transform = get_transform(image_size, crop_size)
    
    fpathlist = []
    for i_subj, subjdir in enumerate(subj_dirs):
        fpathlist += get_fpathlist(args.jpg_root, subjdir, ds_rate=ds_rate)
    
    seq_len = args.seq_len
#     ds_rate = 1
    sample_len = seq_len*ds_rate
    n_samples = 50000
    sample_stride = int(len(fpathlist)/n_samples)
    

    image_paths = [fpathlist[i:i+sample_len:ds_rate] 
     for i in range(0, n_samples*sample_stride, sample_stride)]
# print(len(image_paths[50]))

    dataset = ImageSequenceDataset(image_paths, transform=transform)
    batch_size = 512
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                             num_workers=world_size, persistent_workers=False, collate_fn=None, 
                                             pin_memory=False, prefetch_factor=2)

    input_size = crop_size*crop_size
    hidden_size = args.hidden_size
    lr, dr, pr = 5e-3, 1e-3, 1e-4
    p_sigma = 1./hidden_size
    num_epochs = args.num_epochs
    
    model = RecAutoencoder(input_size, hidden_size, seq_len)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=dr, momentum=0.9)
#     torch.optim.Adam(model.parameters(), lr=lr, weight_decay=dr)
    
    
    train_loader = dataloader

    losses = []
    for epoch in tqdm(range(num_epochs)):
        for data in tqdm(train_loader):
            perturb_model(model, pr, p_sigma)
            images = data
            loss = get_loss(args.objective, model, images)
#             reconstructions = model(images)
#             loss = nn.MSELoss()(reconstructions, images)
            optimizer.zero_grad()
            loss.backward()
            losses.append(loss.item())
            optimizer.step()
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
    
    PATH = args.save_path #'/N/slate/sheybani/tmp_dir/trainedmodels/slowness_ae/rae4951_h16.pt'
    
    torch.save({
            'model_state_dict': model.state_dict(),
            'loss': losses}, PATH)
    
#     torch.save(model, PATH)