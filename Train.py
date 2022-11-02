#!/usr/bin/env python
# coding: utf-8

# In[145]:


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
import glob
import torch.optim as optim


# ### Load the data

# In[170]:


#define a directory which contains images (fruits)
train_path = '../data/'

# Define transformations for training 
transform = transforms.Compose(
    [transforms.Resize((200,200)), #Ensure all images are the same size
    transforms.ToTensor(),  #Transform to be a tensor
    transforms.Normalize(torch.Tensor((0.5, 0.5, 0.5)), torch.Tensor((0.5, 0.5, 0.5)))])



#Extract data
train_data =torchvision.datasets.ImageFolder(train_path,transform=transform)
print(f'The total number of training images after preprocessing: {len(train_data)}')


# In[171]:


#Classes
import pathlib
root=pathlib.Path(train_path)
classes=sorted([j.name.split('/')[-1] for j in root.iterdir()])
print(f'The three classes of the data set:'+ str(classes))

As the data does not include a test set, the trai data provided will be split 80:20 using random split to test the accuracy of the model created.
Creating a subset of the training set, by default torchvision will apply changes in transforms for both the train and validation set. To prevent this from occuring, a deepcopy of the vaildation will also be created.
# In[172]:


#Define the number of instance that is in each split of the train and val set
v_ratio = 0.8
train_n = int(len(train_data) * v_ratio)
val_n = len(train_data) - train_n

#Split the data
train_set, val_set = data.random_split(train_data,
                                           [train_n, val_n])

print(f'Number of images in the training set: {len(train_set)}')
print(f'Number of images in the validation set: {len(val_set)}')


# In[173]:


import copy
valid_set = copy.deepcopy(val_set)


# In[174]:


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device


# In[175]:


#Data Loader
train_loader = data.DataLoader(train_set, shuffle=True, batch_size=64)
val_loader = data.DataLoader(valid_set, batch_size=64)


# # Multi Layer Perceptron 
# 

# In[192]:



class MultiLayerPerceptron(nn.Module):
    
    def __init__(self, input_size=200*200*3, num_classes=3):
        #input_size: 200x200 size of images, 3-RBG
        #Output_size=3 classes
        super().__init__()
        self.fc1 = nn.Linear(input_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

        
        #the forward function
    def forward(self, x):
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = (self.fc3(x))
        

        return x


mlp_model = MultiLayerPerceptron()
print(mlp_model)


# In[185]:


### Define Loss FUnction
import torch.optim as optim
mlp_optimizer = optim.Adam(mlp_model.parameters(),lr=0.001)
criterion = nn.CrossEntropyLoss()


# ### Train the MLP Network

# In[186]:


import time

#Calculate execution
start_time = time.time()

#indicates the number of passes of the entire training dataset the machine learning algorithm has completed
epochs = 5
running_loss = 0.0

for i in range(epochs):
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        mlp_optimizer.zero_grad()

        # forward + backward + optimize
        outputs = mlp_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        mlp_optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0
            
print('Finished Training')
print("--- %s seconds ---" % (time.time() - start_time))


# In[187]:


PATH = './mlpmodel.pth'
torch.save(mlp_model.state_dict(), PATH)


# ## Testing on the validation set

# In[188]:


#Load predicted classes -A good way to compare as I am not able to load the images without disconnecting to the kernel
_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
                              for j in range(10)))


# In[189]:


correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in val_loader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = mlp_model(images)
        
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')


# In[190]:


# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# again no gradients needed
with torch.no_grad():
    for data in val_loader:
        images, labels = data
        outputs = mlp_model(images)
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


# ### CNN Model
# 
# reference: https://www.geeksforgeeks.org/implementation-of-a-cnn-based-image-classifier-using-pytorch/

# ### Model Training
# 

# In[ ]:


#Calculate the size of train_set and val_set
t_count = len(train_set)
v_count = len(val_set)

print(f'Length of train_set:' +str(t_count))
print(f'Length of val_set :' +str(v_count))


# ### Defining the Model

# In[177]:


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            
            #Shape = (batch size, RBG, Height, width) = 64,3,200,200 
            
        #First Layer
            #Input = 3 x 200,200, Output = 12 x 12 x 12
            nn.Conv2d(in_channels = 3, out_channels = 12, kernel_size = 3, padding = 1), #shape=(64,12,200,200)
            nn.ReLU(),
            
            

            nn.MaxPool2d(kernel_size=2), 
            #Resize the image by a factor of 2
            # shape (64, 12, 100,100)
            
        #Second Layer
  
            
            nn.Conv2d(in_channels = 12, out_channels = 32, kernel_size = 3, padding = 1),
            
            #shape (64, 32, 100,100)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            # shape (64, 32, 50,50)
    
        #Third Layer
            
            nn.Conv2d(in_channels = 32, out_channels = 12, kernel_size = 3, padding = 1), # shape (64,32,50,50)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # shape (64,12, 25,25)
  
            nn.Flatten(),
            nn.Linear(12*25*25, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )
  
    def forward(self, x):
        return self.model(x)

    
net= Net()
print(net)


# In[179]:


import time

#Calculate execution
start_time = time.time()


#Selecting the appropriate training device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Net().to(device)
  
#Defining the model hyper parameters
num_epochs = 5
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
  
#Training process begins
train_loss_list = []
for epoch in range(num_epochs):
    print(f'Epoch {epoch+1}/{num_epochs}:', end = ' ')
    train_loss = 0
      
    #Iterating over the training dataset in batches
    net.train()
    for i, (images, labels) in enumerate(train_loader):
          
        #Extracting images and target labels for the batch being iterated
        images = images.to(device)
        labels = labels.to(device)
  
        #Calculating the model output and the cross entropy loss
        outputs = net(images)
        loss = criterion(outputs, labels)
  
        #Updating weights according to calculated loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()


    #Printing loss for each epoch
    train_loss_list.append(train_loss/len(train_loader))
    print(f"Training loss = {train_loss_list[-1]}")   

print('Finished Training')
print("--- %s seconds ---" % (time.time() - start_time))


# In[191]:


import time

#Calculate execution
start_time = time.time()


#Selecting the appropriate training device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Net().to(device)
  
#Defining the model hyper parameters
num_epochs = 10
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
  
#Training process begins
train_loss_list = []
for epoch in range(num_epochs):
    print(f'Epoch {epoch+1}/{num_epochs}:', end = ' ')
    train_loss = 0
      
    #Iterating over the training dataset in batches
    net.train()
    for i, (images, labels) in enumerate(train_loader):
          
        #Extracting images and target labels for the batch being iterated
        images = images.to(device)
        labels = labels.to(device)
  
        #Calculating the model output and the cross entropy loss
        outputs = net(images)
        loss = criterion(outputs, labels)
  
        #Updating weights according to calculated loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()


    #Printing loss for each epoch
    train_loss_list.append(train_loss/len(train_loader))
    print(f"Training loss = {train_loss_list[-1]}")   

print('Finished Training')
print("--- %s seconds ---" % (time.time() - start_time))


# In[181]:


PATH2 = './model.pth'
torch.save(net.state_dict(), PATH2)


# In[180]:


#Plotting loss for all epochs
plt.plot(range(1,num_epochs+1), train_loss_list)
plt.xlabel("Number of epochs")
plt.ylabel("Training loss")


# ### Test the model

# In[193]:


correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in val_loader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = net(images)
        
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')


# In[183]:


# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# again no gradients needed
with torch.no_grad():
    for data in val_loader:
        images, labels = data
        outputs = net(images)
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

