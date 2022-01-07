# Import the required libraries
import pandas as pd
from PIL import Image
import torch
import torchvision


### Load and Transform the Images

## Variable settings (update as needed)
# File paths
base_path = '' # Update this if data files are in a different folder
img_path = base_path + 'train/'

# Image size
img_res = 224 # Update this as needed in your model

# Transformations to be done to the image
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((img_res, img_res)),
    torchvision.transforms.ToTensor()
])

transform_flip = torchvision.transforms.Compose([
    torchvision.transforms.Resize((img_res, img_res)),
    torchvision.transforms.RandomHorizontalFlip(p = 1.0),
    torchvision.transforms.ToTensor()
])

transform_blur = torchvision.transforms.Compose([
    torchvision.transforms.Resize((img_res, img_res)),
    torchvision.transforms.GaussianBlur(kernel_size = (5, 5), sigma=0.3),
    torchvision.transforms.ToTensor()
])

transform_crop = torchvision.transforms.Compose([
    torchvision.transforms.Resize((img_res, img_res)),
    torchvision.transforms.CenterCrop(int(0.9*img_res)),
    torchvision.transforms.Resize((img_res, img_res)),
    torchvision.transforms.ToTensor()
])


# Read the training data
meta_train = pd.read_csv(base_path+'train.csv')


# Create list to store the data
data = []


# Load and transform the images/metadata then populate the created list
print('Processing images...')
for idx, row in meta_train.iterrows():    
    # Get elements from dataframe
    img_name = row['Id']+'.jpg'
    meta_data = row.iloc[1:-1]
    pawpularity_score = row['Pawpularity']
    
    # Perform transformations on the image
    img = Image.open(img_path+img_name)
    img_resize = transform(img)
    
    # Output is a tuple containing: (1) image data, (2) meta data, (3) pawpularity score
    data.append((img_resize, torch.tensor(meta_data), torch.tensor([pawpularity_score])))
    
    # Apply data augmentation
    # Add flipped images
    if (pawpularity_score <= 20) | (pawpularity_score >= 50):
        img_flip = transform_flip(img)
        data.append((img_flip, torch.tensor(meta_data), torch.tensor([pawpularity_score])))
    
    # Add slightly blurred images and center-cropped images
    if (pawpularity_score <= 10) | (pawpularity_score >= 60):
        img_blur = transform_blur(img)
        data.append((img_blur, torch.tensor(meta_data), torch.tensor([pawpularity_score])))
        img_crop = transform_crop(img)
        data.append((img_crop, torch.tensor(meta_data), torch.tensor([pawpularity_score])))
    
    # Print status update of data transformation
    if ((idx + 1) % 500 == 0) | (idx + 1 == len(meta_train)): # Print update every 500th image and when all images have been processed
        print(f'\t{idx + 1} out of {len(meta_train)} images processed!')

print('\n')
print(f'Complete! There are now a total of {len(data)} images!')


# Save the data for easy loading in the future
torch.save(data, 'img_multiaugment_'+str(img_res)+'.pt')