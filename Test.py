#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
get_ipython().run_line_magic('matplotlib', 'inline')
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.utils.data as data
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable

import torch.optim as optim
import pathlib


# ### Loading the Test data

# In[ ]:



test_path = '../test_data'

# Define transformations for training 
transform = transforms.Compose(
    [transforms.Resize((200,200)), #Ensure all images are the same size
    transforms.ToTensor(),  #Transform to be a tensor
    transforms.Normalize(torch.Tensor((0.5, 0.5, 0.5)), torch.Tensor((0.5, 0.5, 0.5)))])



#Extract data
test_data =torchvision.datasets.ImageFolder(test_path,transform=transform)
print(f'The total number of training images after preprocessing: {len(test_data)}')


# In[ ]:


#Classes
root=pathlib.Path(test_path)
classes=sorted([j.name.split('/')[-1] for j in root.iterdir()])
print(f'The three classes of the data set:'+ str(classes))


# In[ ]:


#Data Loader
test_loader = data.DataLoader(test_data, batch_size=64)


# ### Convolutional Neural Network

# In[5]:


model = torch.load("../model.pth")


# In[ ]:


correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = model(images)
        
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')


# In[ ]:


# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# again no gradients needed
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = model(images)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1


# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')


# In[ ]:




