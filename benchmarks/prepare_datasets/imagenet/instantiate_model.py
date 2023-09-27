import sys, os

env_root = '/N/project/baby_vision_curriculum/pythonenvs/hfenv/lib/python3.10/site-packages/'
sys.path.insert(0, env_root)


import numpy as np
import torch, torchvision
from torchvision import transforms as tr
# from torch import nn
# from torch.nn import functional as F
import os
# import random
# import time
from tqdm import tqdm
from pathlib import Path
# import math
import argparse
import pandas as pd
import warnings
import matplotlib.pyplot as plt

import transformers

from copy import deepcopy


def get_config(image_size, args):
    arch_kw = args.architecture
    if arch_kw=='small2':
        hidden_size = 768
        intermediate_size = 4*768
        num_attention_heads = 6
        num_hidden_layers = 6
        
        config = transformers.VideoMAEConfig(image_size=image_size, patch_size=16, num_channels=3,
                                             num_frames=16, tubelet_size=2, 
                                             hidden_size=hidden_size, num_hidden_layers=num_hidden_layers, num_attention_heads=num_attention_heads,
                                             intermediate_size=intermediate_size)
    else:
        raise ValueError
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

class Args:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
            
model_root = '/N/project/baby_vision_curriculum/trained_models/generative/v2/s2/'

chpt_name = 'model_g1_seed_1133_other_1345_mask50_small2_pre.g0.pt'
# 'model_g1_seed_1133_other_1345_mask50_small2_pre.g2.pt'

chpt_path = model_root+chpt_name

image_size = 224
architecture='small2'
init_checkpoint_path=chpt_path
args = Args(architecture=architecture, init_checkpoint_path=init_checkpoint_path)

xmodel = get_model(image_size, args)

if args.init_checkpoint_path!='na':
    print('args.init_checkpoint_path:',args.init_checkpoint_path)
    # initialize the model using the checkpoint
    xmodel = init_model_from_checkpoint(xmodel, args.init_checkpoint_path)
        