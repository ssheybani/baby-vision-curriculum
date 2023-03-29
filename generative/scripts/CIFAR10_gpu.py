#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
if 'Carbonate' in os.getcwd().split('/'):
    sys.path.insert(0,'/N/slate/hhansar/hgenv/lib/python3.10/site-packages')


# In[2]:


import torch
import torchvision
import torchvision.transforms as transforms
from transformers import AutoImageProcessor


# In[3]:


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Assuming that we are on a CUDA machine, this should print a CUDA device:

print(device)


# In[4]:


transform = transforms.Compose(
#     [transforms.Resize((16, 16)),
     [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 8

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=20)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=20)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# In[5]:


# import matplotlib.pyplot as plt
# import numpy as np

# # functions to show an image


# def imshow(img):
#     img = img / 2 + 0.5     # unnormalize
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()


# # get some random training images
# dataiter = iter(trainloader)
# images, labels = next(dataiter)

# # show images
# imshow(torchvision.utils.make_grid(images))
# # print labels
# print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))


# In[6]:


import torch.nn as nn
import torch.nn.functional as F


from transformers import AutoImageProcessor, VideoMAEForPreTraining, VideoMAEConfig, VideoMAEModel

config = VideoMAEConfig()
model = VideoMAEModel(config)
model.eval()
print(model)
# backbone = torch.load('encoder.pt')
# backbone.eval()# class Net(nn.Module):

num_classes = 10
class CIFAR10Benchmark(nn.Module):
    def __init__(self, backbone):
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
CIFAR10Model = CIFAR10Benchmark(model)
CIFAR10Model = CIFAR10Model.to(device)


# In[7]:


import torch.optim as optim
from tqdm import tqdm
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(CIFAR10Model.parameters(), lr=0.001, momentum=0.9)
image_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")


# In[8]:


for epoch in range(30):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(tqdm(trainloader), 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = list(inputs)
        inputs = image_processor(inputs, return_tensors="pt").pixel_values
#         print(inputs.shape)
        inputs = torch.swapaxes(inputs, 0,1)
        num_frames = 16
        inputs = inputs.repeat(1,num_frames, 1, 1,1)
#         print(inputs.shape)
        inputs, labels = inputs.to(device), labels.to(device)
#         print(inputs.shape)

#         # zero the parameter gradients
        optimizer.zero_grad()
#         print(inputs.device)
        # forward + backward + optimize
        outputs = CIFAR10Model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 10 == 0:    # print every 2000 mini-batches
            print(f'[{i}] loss: {running_loss / 10:.3f}')
            running_loss = 0.0
    print(f'Epoch loss {epoch}',running_loss/ len(trainloader))

print('Finished Training')


# In[ ]:


nvidia-smi


# In[ ]:


PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)

dataiter = iter(testloader)
images, labels = next(dataiter)
images = list(images)
images = image_processor(images, return_tensors="pt").pixel_values
#         print(inputs.shape)
images = torch.swapaxes(images, 0,1)
num_frames = 16
images = images.repeat(1,num_frames, 1, 1,1)
#         print(inputs.shape)
images, labels = images.to(device), labels.to(device)
# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))

net = CIFAR10Benchmark(model)
net.load_state_dict(torch.load(PATH))
net.to(device)

outputs = net(images)



_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
                              for j in range(4)))


correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images = list(images)
        images = image_processor(images, return_tensors="pt").pixel_values
        images = torch.swapaxes(images, 0,1)
        num_frames = 16
        images = images.repeat(1,num_frames, 1, 1,1)
        images, labels = images.to(device), labels.to(device)
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')