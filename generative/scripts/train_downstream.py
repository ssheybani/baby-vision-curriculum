import sys, os


if 'BigRed200' in os.getcwd().split('/'):
#     print('Running on BigRed200')
    sys.path.insert(0,'/geode2/home/u080/sheybani/BigRed200/spenv/lib/python3.10/site-packages')

# SCRIPT_DIR = os.getcwd() #os.path.realpath(os.path.dirname(inspect.getfile(inspect.currentframe())))
# util_path = os.path.normpath(os.path.join(SCRIPT_DIR, '..', 'util'))
# sys.path.insert(0, util_path)    

# import time
import torch
from tqdm import tqdm
from pathlib import Path
import math
from copy import deepcopy

import torch.distributed as dist
from ddputils import is_main_process, save_on_master, setup_for_distributed

import pandas as pd
import numpy as np

def train_classifier(model, dataloaders, criterion, optimizer, device,
                num_epochs=25, scheduler=None, verbose=True):
    """
    Generic PyTorch script for training a model on a dataloader with 
    input and target.
    
    Arguments
    ---
    model: Torch nn model
        For linear classification, the parameters must be frozen before.
    
    dataloaders: dict of two dataloaders named 'train', 'val'
    
    optimizer: Torch optimizer
    
    criterion: a Torch loss
        receives outputs and labels.
    
    
    Returns
    ---
    val_acc_history: list
        validation accuracy history
    
    
    Usage
    ---

    xmodel, hist = train_model(xmodel, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs)

    """
    
    
    # since = time.time()
    
    train_acc_history = []
    val_acc_history = []
    
    best_model_wts = deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        if verbose:
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
            if verbose:
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc.cpu().item())
            else:
                train_acc_history.append(epoch_acc.cpu().item())

        print()
        if scheduler is not None:
            scheduler.step()

    # time_elapsed = time.time() - since
    print('Training complete')#' in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, [train_acc_history, val_acc_history]


def train_classifier_ddp(world_size, rank, model, dataloaders, criterion, optimizer, 
                         results_fpath, num_epochs=25, scheduler=None, 
                         verbose=True):
    """
    Generic PyTorch script for training a model on a dataloader with 
    input and target.
    
    Arguments
    ---
    model: Torch nn model
        For linear classification, the parameters must be frozen before.
    
    samplers:
        dict of two distributed samplers named 'train', 'val'
    dataloaders: 
        dict of two dataloaders named 'train', 'val'
    
    optimizer: Torch optimizer
    
    criterion: a Torch loss
        receives outputs and labels.
    
    results_fpath: path to the CSV file to store the results
    
    Returns
    ---
    None
    Stores a csv file in results_fpath

    """
    
    verbose = (verbose and is_main_process())
    
    # data = {
    #     'results': {'train_acc':[],
    #                 'val_acc':[]}
    # }

    # outputs = [None for _ in range(world_size)]
    
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
            dataloaders[phase].sampler.set_epoch(i_ep)
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = torch.tensor([0.0], device=rank)
            running_corrects = torch.tensor([0.0], device=rank)

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase]):

                inputs = inputs.to(rank)
                labels = labels.to(rank)
#                 print(rank, 'labels',labels)
#                 print('inputs shape, device: ', inputs.shape, inputs.device)
#                 print('labels shape, device: ', labels.shape, labels.device)
#                 print('model device: ', model.module.fc.weight.device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
#                 print(rank, 'labels shape, device: ', labels.shape, labels.data.device)
#                 print(rank, 'preds shape, device: ', preds.shape, preds.device)
                running_corrects += torch.sum(preds == labels.data)
            
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
        if scheduler is not None:
            scheduler.step()
    if verbose:
        print('Training complete')
        print('Best val Acc: {:4f}'.format(best_acc.item()))

    results_df = pd.DataFrame({'epoch': np.arange(len(train_acc_history)),
                               'train_acc':train_acc_history,
                               'val_acc': val_acc_history})

    if is_main_process():
        results_df.to_csv(results_fpath, sep=',', float_format='%.3f')
        # np.savetxt(outfpath, np.asarray(acc_history).T, delimiter=',', fmt='%.3f')
    