#!/usr/bin/env python

## Importing required libraries

from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)



## Setting code to run on CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# #### Loading the data - Input images after Data Processing


# Images scaled to 224x224 pixels
data = torch.load('img_multiaugment_224.pt')   ## ALL IMAGES


# #### Splitting data into Test and Validation sets and applying the Saliency Map on Validation Images


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


# ### Initialize the model

# Takes in a pretrained model passed through additional FC layers and concatenates this to feature embeddings
# of the metadata passed through FC layers

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
        # Process the image data
        img_layer = self.pretrained_model(img)
        img_layer = self.added_layers(img_layer)

        # Process the meta data
        meta_layer = self.meta_layers(meta)

        # Concatenate the image and meta data embeddings and pass through final layers
        final_layer = torch.cat([img_layer, meta_layer], axis = 1)
        final_layer = self.final_layers(final_layer)

        return final_layer


# Initialize our pretrained model (DenseNet)
densenet = models.densenet161(pretrained=True)

param_idx = 0
# Freeze the first parameters of the pretrained model
for param in densenet.parameters():
    param_idx += 1
    if param_idx <= 400:
        param.requires_grad = False

# Redefine the final densenet layer
densenet.classifier = nn.Linear(2208, 1024)


# #### Use the trained model (Best Model : Densenet)

paw_model = torch.load(('best_densenet.pth'), map_location=torch.device('cpu'))
paw_model.eval()


# ### Processing Validation images for Saliency Map Visualization

imgs = []
meta_lst = []
labels_lst = []
names_lst = []
for (inputs, meta, labels, name) in dataloaders_dict['train']:
    inputs = inputs.to(device)
    meta = meta.float().to(device)
    labels = labels.float().to(device)

    for i in range(inputs.shape[0]):
        new_img = inputs[i,:,:,:].reshape((1,inputs.shape[1], inputs.shape[2], inputs.shape[3]))
        imgs.append(new_img)

        meta_lst.append(meta[i].reshape((1, meta[i].shape[0])))
        labels_lst.append(labels[i])
        names_lst.append(name[i])


# ### Saving Saliency Maps for all Validation Images

# ### Generating Saliency Maps based on scores

# #### High Score

#high_score_imgs = [(idx, score[0], score[1]) for idx,score in enumerate(zip(labels_lst, names_lst)) if score[0] >= 80]
#img_bucket = high_score_imgs


## Images with high score (80 - 100)
high_score_imgs = [(idx,score) for idx,score in enumerate(labels_lst) if score >= 80]
img_bucket = high_score_imgs


saliency_dict = {} ## List which will store images and their graidents in a tuple

for idx, i in enumerate(img_bucket):
    # Getting each image
    img = imgs[i[0]]
    meta_model = meta_lst[i[0]]
    img_name = names_lst[i[0]]

    ## Updating gradients required for images
    img = img.to(device)
    img.requires_grad_()

    ## Backproagating gradients with respect to the images
    output = paw_model(img, meta_model)
    output_idx = output.argmax()
    output_max = output[0, output_idx]
    output_max.backward()

    ## Saving Image and its Saliency
    saliency, _ = torch.max(img.grad.data.abs(), dim=1)
    saliency = saliency.reshape(224, 224)

    # Reshape the image
    image = img.reshape(-1, 224, 224)

    saliency_dict[i[0]] = (image, saliency)

    # Saving the image and the saliency map
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(image.cpu().detach().numpy().transpose(1, 2, 0))
    ax[0].axis('off')
    ax[1].imshow(saliency.cpu(), cmap='hot')
    ax[1].axis('off')
    plt.tight_layout()
    #fig.suptitle('The Image and Its Saliency Map')

    fig.savefig('Saliency_Maps/'+str('high_score')+'/img_saliency_'+str(img_name)+'_'+str(i[1].item())+'.png')

    # Print status for every 10 images
    if ((idx + 1) % 10 == 0) | (idx + 1 == len(img_bucket)): # Print update every 500th image and when all images have been processed
        print(f'\t{idx + 1} out of {len(img_bucket)} images processed!')


# #### Low Score

## Images with high score (0 - 20)
low_score_imgs = [(idx,score) for idx,score in enumerate(labels_lst) if score <= 30]
img_bucket = low_score_imgs


saliency_dict = {} ## List which will store images and their graidents in a tuple

for idx, i in enumerate(img_bucket):
    # Getting each image
    img = imgs[i[0]]
    meta_model = meta_lst[i[0]]
    img_name = names_lst[i[0]]

    ## Updating gradients required for images
    img = img.to(device)
    img.requires_grad_()

    ## Backproagating gradients with respect to the images
    output = paw_model(img, meta_model)
    output_idx = output.argmax()
    output_max = output[0, output_idx]
    output_max.backward()

    ## Saving Image and its Saliency
    saliency, _ = torch.max(img.grad.data.abs(), dim=1)
    saliency = saliency.reshape(224, 224)

    # Reshape the image
    image = img.reshape(-1, 224, 224)

    saliency_dict[i[0]] = (image, saliency)

    # Saving the image and the saliency map
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(image.cpu().detach().numpy().transpose(1, 2, 0))
    ax[0].axis('off')
    ax[1].imshow(saliency.cpu(), cmap='hot')
    ax[1].axis('off')
    plt.tight_layout()
    #fig.suptitle('The Image and Its Saliency Map')

    fig.savefig('Saliency_Maps/'+str('low_score')+'/img_saliency_'+str(img_name)+'_'+str(i[1].item())+'.png')

    # Print status for every 10 images
    if ((idx + 1) % 10 == 0) | (idx + 1 == len(img_bucket)): # Print update every 500th image and when all images have been processed
        print(f'\t{idx + 1} out of {len(img_bucket)} images processed!')


# #### Average Score

## Images with high score (0 - 20)
avg_score_imgs = [(idx,score) for idx,score in enumerate(labels_lst) if (score > 30 and score < 80)]
img_bucket = avg_score_imgs


saliency_dict = {} ## List which will store images and their graidents in a tuple

for idx, i in enumerate(img_bucket):
    # Getting each image
    img = imgs[i[0]]
    meta_model = meta_lst[i[0]]
    img_name = names_lst[i[0]]

    ## Updating gradients required for images
    img = img.to(device)
    img.requires_grad_()

    ## Backproagating gradients with respect to the images
    output = paw_model(img, meta_model)
    output_idx = output.argmax()
    output_max = output[0, output_idx]
    output_max.backward()

    ## Saving Image and its Saliency
    saliency, _ = torch.max(img.grad.data.abs(), dim=1)
    saliency = saliency.reshape(224, 224)

    # Reshape the image
    image = img.reshape(-1, 224, 224)

    saliency_dict[i[0]] = (image, saliency)

    # Saving the image and the saliency map
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(image.cpu().detach().numpy().transpose(1, 2, 0))
    ax[0].axis('off')
    ax[1].imshow(saliency.cpu(), cmap='hot')
    ax[1].axis('off')
    plt.tight_layout()
    #fig.suptitle('The Image and Its Saliency Map')

    fig.savefig('Saliency_Maps/'+str('avg_score')+'/img_saliency_'+str(img_name)+'_'+str(i[1].item())+'.png')

    # Print status for every 10 images
    if ((idx + 1) % 10 == 0) | (idx + 1 == len(img_bucket)): # Print update every 500th image and when all images have been processed
        print(f'\t{idx + 1} out of {len(img_bucket)} images processed!')






