#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 20:32:20 2021

@author: shanmukhateja
"""
# Takes in a pretrained model passed through additional FC layers and concatenates this to feature embeddings
# of the metadata passed through FC layers


from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
from PIL import Image
import time
import os
import copy
import cv2
import pandas as pd
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

# PyTorch Modules
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms, datasets
import torchvision.transforms as transforms
from torch.utils.data.dataset import Subset
import torchvision.models as models
import torch.optim as optim
from torchvision.utils import make_grid, save_image


class PawModel(nn.Module):
    def __init__(self, pretrained_model):
        super(PawModel, self).__init__()
        
        # Call the pretrained model and add FC layer
        self.pretrained_model = pretrained_model
#         self.added_layers = nn.Linear(1024, 512)
        
#         # Pass the metadata through FC layers
#         self.meta_layers = nn.Sequential(nn.Linear(12, 24),
#                                          nn.Linear(24, 48),
#                                          nn.Linear(48, 96))
        
#         # Pass the concatenated embeddings through additional FC layers
#         self.final_layers = nn.Sequential(nn.Linear(608, 120),
#                                          nn.Linear(120, 1),
#                                          nn.ReLU())
        self.added_layers = nn.Linear(1024, 1024)
        
        # Pass the metadata through FC layers
        self.meta_layers = nn.Sequential(nn.Linear(12, 128),
                                         nn.Linear(128, 256),
                                         nn.Linear(256, 512),
                                         nn.Linear(512, 1024))
        
        # Pass the concatenated embeddings through additional FC layers
        self.final_layers = nn.Sequential(nn.Linear(2048, 1024),
                                          nn.Dropout(0.2),
                                          nn.Linear(1024, 512),
                                          nn.Dropout(0.2),
                                          nn.Linear(512, 1),
                                          nn.ReLU())
    
    def forward(self, img, meta):
        # img, meta = x
        # Process the image data
        img_layer = self.pretrained_model(img)
        img_layer = self.added_layers(img_layer)
        
        # Process the meta data
        meta_layer = self.meta_layers(meta)
        
        # Concatenate the image and meta data embeddings and pass through final layers
        final_layer = torch.cat([img_layer, meta_layer], axis = 1)
        final_layer = self.final_layers(final_layer)
        
        return final_layer 

#%%
        
# Loading Images scaled to 224x224 pixels
data = torch.load('img_multiaugment_224.pt')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Split the data into 90-10% split
train_ratio = 0.9
train_count, val_count = int(train_ratio * len(data)), len(data) - int(train_ratio * len(data))
train, val = torch.utils.data.random_split(data, (train_count, val_count), generator=torch.Generator().manual_seed(42))
# Set up the train loader (update batch size / other hyperparameters as desired)
train_loader = torch.utils.data.DataLoader(train, batch_size=32, shuffle=True)
# Set up the validation loader (update batch size / other hyperparameters as desired)
val_loader = torch.utils.data.DataLoader(val, batch_size=32, shuffle=True)
# Create training and validation dataloaders dictionary
dataloaders_dict = {'train': train_loader, 'val': val_loader}

#Loading data from validation set 
imgs = []
meta_lst = []
labels_lst = []
for (inputs, meta, labels) in dataloaders_dict['val']:
    inputs = inputs.to(device)
    meta = meta.float().to(device) 
    labels = labels.float().to(device)
    
    for i in range(inputs.shape[0]):
        new_img = inputs[i,:,:,:].reshape((1,inputs.shape[1], inputs.shape[2], inputs.shape[3]))
        imgs.append(new_img)
        
        meta_lst.append(meta[i].reshape((1, meta[i].shape[0])))
        labels_lst.append(labels[i])

#%%

# Images scaled to 224x224 pixels
model = torch.load('best_densenet.pth', map_location=torch.device('cpu'))
# model.eval()

# img = imgs[123]
# meta_model = meta_lst[123]
# model(img, meta_model)

# plt.imshow(img.detach().numpy().squeeze().transpose(1,2,0))
# print("Pawpularity Score of the image is ",model(img,meta_model))


#%%
#GRADCAM Manual
class grad_cam_class():

    def grad_cam(self, X_tensor, meta, y_tensor, gc_model):
            from PIL import Image
            """
            Input:
            - X_tensor: Input images; Tensor of shape (N, 3, H, W)
            - y: Labels for X; LongTensor of shape (N,)
            - model: A pretrained CNN that will be used to compute the gradcam.
            """
            conv_module = model.pretrained_model.features.denseblock4.denselayer24.conv2
            self.gradient_value = None  # Stores gradient of the module you chose above during a backwards pass.
            self.activation_value = None  # Stores the activation of the module you chose above during a forwards pass.
    
            def gradient_hook(a, b, gradient):
                self.gradient_value = gradient[0]
    
            def activation_hook(a, b, activation):
                self.activation_value = activation
    
            conv_module.register_forward_hook(activation_hook)
            conv_module.register_backward_hook(gradient_hook)

            ##############################################################################
# GradCAM computes the gradients of the target output with respect to the given layer, averages for each output channel (dimension 2 of output), and multiplies the average gradient for each channel by the layer activations. The results are summed over all channels.
            ##############################################################################
            
            scores = gc_model(X_tensor, meta)      
            # scores_out = scores[torch.arange(len(scores)), y_tensor]
            loss = torch.sum(scores)
            mseloss = nn.MSELoss()
            loss = mseloss(scores, y_tensor)
            print("###############")
            # print(scores, y_tensor, loss)
            
            loss.backward()
    
            pooled_gradients = torch.mean(self.gradient_value, dim=[0, 2, 3])
            print("###############")
            # print(self.activation_value.shape)
            # print(pooled_gradients.shape)
            
            Chs = self.activation_value.shape[1]
            print("###############")
            # print(Chs)
            
            for channel in range(Chs):
                self.activation_value[:, channel, :, :] *= pooled_gradients[channel]
            
            cam = torch.mean(self.activation_value, dim=1).squeeze()
            print("###############")
            # print(cam)
            cam = np.maximum(cam.detach().numpy(), 0)
            print("###############")
            # print(cam)
            
          # Rescale GradCam output to fit image.
            cam_scaled = []
            size_img = X_tensor[0, 0, :, :].shape
            cam_channel = cam.shape[0]
            #upsampling
            cam_scaled =np.array(Image.fromarray(cam).resize(size_img)) 
            
            cam2 = np.array(cam_scaled)
            cam2 -= np.min(cam2)
            cam2 /= np.max(cam2)
            # return cam2, self.activation_value, pooled_gradients

            cam_acv= self.activation_value
            cam_pg = pooled_gradients

            
            result = torch.zeros((48,7,7)) 
            for i in range(48):
                result[i] = cam_pg[i]*cam_acv[0,i,:,:]
            result2 = torch.mean(result,dim = 0).squeeze()
            
            result3= np.maximum(result2.detach().numpy(), 0)
            
            result4=np.array(Image.fromarray(result3).resize((224,224)))
            result4 = np.array(result4)
            result4 -= np.min(result4)
            result4 /= np.max(result4)
            return (result4)
       

#%%

avg_score = np.array(['2a5d96b0a4b8b4696f2c2b98b80ebecd',
       '8994f9681ad1bf7b6f200d8440e42f2a',
       '0754b1be915ca4e86ee14eb193f8bc01',
       'f35e594382b4a72a745d2f726d8b0d92',
       'a2c1f071b504550289f9116f52968548',
       '95fdd67ad6542c122f98fb5376840396',
       'b95c7e2f30ea95255abcdc5beab41ba8',
       'e31f170cff01995dc926a3559f412619',
       'ca2689476a4482ed801d07824b561df7',
       'b77cf1e752da7f659edbcefb3ff553ed',
       '8036d7c5d74c67ce7432b47c53e59d37',
       'a188cfda78e8c375bd1909ed71c0ac51',
       '733a827b0d04f612c166e5defd27b4ae',
       '7b4f8a8d9ad9920e5143d0e4c71aaa87',
       '85d2b9e0d2373a6e7de76d4339158588',
       'd0f5effb2a889d38171fdf9cc732789e',
       'fdc1d844dcf01dcb6c1e6affe52b7077',
       'ba1b7354c33a6789d97190384b7d9a10',
       '3c0c585acda38ff8acf8fa00f11ee739',
       '8aafa5da0d3ad53b856446b0afeb68ad',
       '76086f4b38e57d9f15b642ec21ef0295',
       'a06995f3c4cb63b6db4a584e60d361a0',
       '90d77c5fc6d3d180e0f5d8aa75178b28',
       '696beb3baa43f97ae9ef44a53b2dbd42',
       '83cbd41c6a606fececdc7dbe04fb95ee',
       'e375d5fed8249e56e71dbc6676405486',
       '6fbfbb7602dfa549adaac0b71918dd38',
       'd248ce88d89b7825475527b2c99625e1',
       'de461a67e80fce995eec3d21f0d4c1fe',
       'eda539fc68cb39c98dc3bed4bc835fdb'])

for i,name in enumerate(avg_score):

    print(i)
    filepath = '/Users/shanmukhateja/Documents/Teja_Gdrive/GTech/2021 Fall/CS 7643 Deep Learning/Project Cuteness/data/train/' + name + '.jpg'
    train_img = cv2.imread(filepath)
    train_img = cv2.cvtColor(train_img, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(train_img, (224, 224), interpolation = cv2.INTER_AREA)  
    image_resized = torch.tensor(image_resized).clone().detach()
    image_resized = image_resized.permute((2,0,1)).float()/255
    image_resized = image_resized[None,:].float()
    
    
    meta_data = pd.read_csv("/Users/shanmukhateja/Documents/Teja_Gdrive/GTech/2021 Fall/CS 7643 Deep Learning/Project Cuteness/data/train.csv")
    meta = torch.tensor(np.array((meta_data[meta_data["Id"]==name].iloc[:,[1,2,3,4,5,6,7,8,9,10,11,12]])))
    meta = meta.float()  
    pop = model(torch.tensor(image_resized),meta)

    y_tensor = torch.tensor(np.array((meta_data[meta_data["Id"]==name].iloc[:,13])))
    print("###############")
    print(pop)
    print("###############")
    print(y_tensor)

    img = torch.tensor(image_resized)
    meta_model = meta
    y_tensor = y_tensor.float()
       
    #Gradcam
    gcclass= grad_cam_class()            
    result_out= gcclass.grad_cam(img,meta,y_tensor,model)
    # plt.imshow(img.detach().numpy().squeeze().transpose(1, 2, 0))
    # plt.imshow(result_out)

    
    # Visualize the image and the Gradcam map
    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(img.detach().numpy().squeeze().transpose(1, 2, 0))
    ax[0].axis('off')
    ax[1].imshow(result_out)
    ax[1].axis('off')
    ax[2].imshow(img.detach().numpy().squeeze().transpose(1, 2, 0))
    ax[2].imshow(result_out,alpha=0.5)
    ax[2].axis('off')
    plt.tight_layout()
    # fig.suptitle('The Image and Its Gradcam Map')
    plt.show()
    fig.savefig('/Users/shanmukhateja/Documents/Teja_Gdrive/GTech/2021 Fall/CS 7643 Deep Learning/Project Cuteness/Results/Avg/' +'fuse_' + str(int(y_tensor))+'_'+str(int(model(img,meta_model))) + 'val'+ str(name)+'.jpg' )
        
#%%
    

low_score = np.array(['4bd8a698f303ce384ae7f94486c97135',
       '73081589e58eb4e91ba8d516b2eab818',
       'ce3c31f422fe79436b1380d3896d0e02',
       '7a6963139dfe7bad4652e5dd560c5b31',
       '6cbca73964b22d569c82180dfab46fe0',
       'f300acf68cd04f4fca3f8d2eecc887b4',
       '9f71e987512b798f4bc196718d7eb1d9',
       '2969cc601f173221a528f8c05571112b',
       'b1839a5890604a26e674ba536a8991f9',
       'dbdbf117c9b2a028ce5773b411720dfc',
       '92740e4e44c2572a43199931789e7166',
       '903c9d54c024b4575b7b360890c444be',
       '803bb1db4f00079a0be91f3319de78f9',
       '50ea6ec9af58dd9f0966e1eae7436ebe',
       '4f27ceb2baf1798ce4f14bd9a290a82c',
       '76420f02afab76d2a6eab95efc816347',
       '11d3ad09f09acf422036ab8368fe44db',
       'd89d171190eb616613632f0180aecb70',
       'd9896c444d85e0882de361ce6bfaebcd',
       '9c01b1f35a946d201aa5cfd761ccb1d6',
       '2ac52920858c86b16642b8dbf44bfb99',
       '93ea840c44849e17e7d876ed772aedfa',
       'b8f920f44800ee71e4659dea84bc9bef',
       'cacd6f1bc88fa1e1b080f0875d72089b',
       '7ba10254b64e21dad13f08311bd288b7',
       '07df2feb88882f2e240306c763d15f26',
       'b7ea2b79695d0b8fb70f734eae86b0da',
       '1fe5b8f82e7e8c0fa0cf0ff5bada400c',
       '14e8fc593665bb46f918421461b64c74',
       '58ca6ecdca7940fac254da35650c1908'])

for i,name in enumerate(low_score):

    print(i)
    filepath = '/Users/shanmukhateja/Documents/Teja_Gdrive/GTech/2021 Fall/CS 7643 Deep Learning/Project Cuteness/data/train/' + name + '.jpg'
    train_img = cv2.imread(filepath)
    train_img = cv2.cvtColor(train_img, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(train_img, (224, 224), interpolation = cv2.INTER_AREA)  
    image_resized = torch.tensor(image_resized).clone().detach()
    image_resized = image_resized.permute((2,0,1)).float()/255
    image_resized = image_resized[None,:].float()
    
    
    meta_data = pd.read_csv("/Users/shanmukhateja/Documents/Teja_Gdrive/GTech/2021 Fall/CS 7643 Deep Learning/Project Cuteness/data/train.csv")
    meta = torch.tensor(np.array((meta_data[meta_data["Id"]==name].iloc[:,[1,2,3,4,5,6,7,8,9,10,11,12]])))
    meta = meta.float()  
    pop = model(torch.tensor(image_resized),meta)

    y_tensor = torch.tensor(np.array((meta_data[meta_data["Id"]==name].iloc[:,13])))
    print("###############")
    print(pop)
    print("###############")
    print(y_tensor)

    img = torch.tensor(image_resized)
    meta_model = meta
    y_tensor = y_tensor.float()
       
    #Gradcam
    gcclass= grad_cam_class()            
    result_out= gcclass.grad_cam(img,meta,y_tensor,model)
    # plt.imshow(img.detach().numpy().squeeze().transpose(1, 2, 0))
    # plt.imshow(result_out)

    
    # Visualize the image and the Gradcam map
    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(img.detach().numpy().squeeze().transpose(1, 2, 0))
    ax[0].axis('off')
    ax[1].imshow(result_out)
    ax[1].axis('off')
    ax[2].imshow(img.detach().numpy().squeeze().transpose(1, 2, 0))
    ax[2].imshow(result_out,alpha=0.5)
    ax[2].axis('off')
    plt.tight_layout()
    # fig.suptitle('The Image and Its Gradcam Map')
    plt.show()
    fig.savefig('/Users/shanmukhateja/Documents/Teja_Gdrive/GTech/2021 Fall/CS 7643 Deep Learning/Project Cuteness/Results/Low/' +'fuse_' + str(int(y_tensor))+'_'+str(int(model(img,meta_model))) + 'val'+ str(name)+'.jpg' )
 
    #%%

high_score = np.array(['5685ddaa23cd1d9726892e9e128bf613',
       'ceeb6ddfdeae718eafd511fad7a16fc7',
       '7279df64b5cc81da8e1ed86dbf2ef013',
       '874605f64bce7a2dddc620f382c93ed8',
       'a1304d3c352b23942b6e40bc3ad28ed8',
       '7c0653c3898abe6a69d8b12e43c7b309',
       'c5d8f1dc9a16bbce7d6e0f9337e27934',
       '1d62cd9439bb452390ece13419a1861e',
       'ad1362089c0b9e4a50ea15e2219c2907',
       '38f19ceba149ba0bbc1602211e24c14b',
       'b694201cfab47e39629a11c31c4d10bb',
       '4321c54ffa2603eafe3a9e5a554dde3e',
       '94b2e52746fc017d27ca5f760e6b17eb',
       'afbbb6b6c6a9a49b88d06eda5b74df69',
       '874476d1929aff73c29132245169067b',
       'c442030c2cb499fea349a86768c56db9',
       'dd004ba6daf8991c0598de6b40b66b22',
       'c1e5610529465ff5b9e337f205b7209a',
       '1f53b8750032a6717212ec1956862d1f',
       '5bc3f6d8b348699d587c6479d8ed5ef5',
       'd87810f762081dd515cdf4d189c209b5',
       '019e7f286c9a3bd4d5a66b662b9465ee',
       '99fac9ce233062bb1a9e0b3350d133ac',
       'e74a5d90d6d62720f2d7ca41b71a553e',
       '0254f54b148543442373d5aad45b2d1a',
       '1174178816648eec4473dff8d35ead04',
       'a271899e5a2c6ab13dcb6df7972e9947',
       '42a900a5e064374ad4fcd50e902e388a',
       '14c6725fb7ca7a16ef64a3da47ab0f65',
       '33857326da9557ee620365ef9d61c68f'])

for i,name in enumerate(high_score):

    print(i)
    filepath = '/Users/shanmukhateja/Documents/Teja_Gdrive/GTech/2021 Fall/CS 7643 Deep Learning/Project Cuteness/data/train/' + name + '.jpg'
    train_img = cv2.imread(filepath)
    train_img = cv2.cvtColor(train_img, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(train_img, (224, 224), interpolation = cv2.INTER_AREA)  
    image_resized = torch.tensor(image_resized).clone().detach()
    image_resized = image_resized.permute((2,0,1)).float()/255
    image_resized = image_resized[None,:].float()
    
    
    meta_data = pd.read_csv("/Users/shanmukhateja/Documents/Teja_Gdrive/GTech/2021 Fall/CS 7643 Deep Learning/Project Cuteness/data/train.csv")
    meta = torch.tensor(np.array((meta_data[meta_data["Id"]==name].iloc[:,[1,2,3,4,5,6,7,8,9,10,11,12]])))
    meta = meta.float()  
    pop = model(torch.tensor(image_resized),meta)

    y_tensor = torch.tensor(np.array((meta_data[meta_data["Id"]==name].iloc[:,13])))
    print("###############")
    print(pop)
    print("###############")
    print(y_tensor)

    img = torch.tensor(image_resized)
    meta_model = meta
    y_tensor = y_tensor.float()
       
    #Gradcam
    gcclass= grad_cam_class()            
    result_out= gcclass.grad_cam(img,meta,y_tensor,model)
    # plt.imshow(img.detach().numpy().squeeze().transpose(1, 2, 0))
    # plt.imshow(result_out)

    
    # Visualize the image and the Gradcam map
    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(img.detach().numpy().squeeze().transpose(1, 2, 0))
    ax[0].axis('off')
    ax[1].imshow(result_out)
    ax[1].axis('off')
    
    ax[2].imshow(img.detach().numpy().squeeze().transpose(1, 2, 0))
    ax[2].imshow(result_out,alpha=0.5)
    ax[2].axis('off')
    
    plt.tight_layout()
    # fig.suptitle('The Image and Its Gradcam Map')
    plt.show()
        
    fig.savefig('/Users/shanmukhateja/Documents/Teja_Gdrive/GTech/2021 Fall/CS 7643 Deep Learning/Project Cuteness/Results/High/' +'fuse_' + str(int(y_tensor))+'_'+str(int(model(img,meta_model))) + 'val'+ str(name)+'.jpg' )
 
    
