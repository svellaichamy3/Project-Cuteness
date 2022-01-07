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


# Read the training data
meta_train = pd.read_csv(base_path+'train.csv')


# Create list to store the data
data = []


# Load and transform the images/metadata then populate the created list
print('Processing images...')
for idx, row in meta_train.iterrows():    
    # Get elements from dataframe
    img_name = row['Id']+'.jpg'
    meta_data = torch.tensor(row.iloc[1:-1])
    pawpularity_score = torch.tensor([row['Pawpularity']])
    
    # Perform transformations on the image
    img = Image.open(img_path+img_name)
    img = transform(img)
    
    # Output is a tuple containing: (1) image data, (2) meta data, (3) pawpularity score
    data.append((img, meta_data, pawpularity_score))
    
    # Print status update of data transformation
    if ((idx + 1) % 500 == 0) | (idx + 1 == len(meta_train)): # Print update every 500th image and when all images have been processed
        print(f'{idx + 1} out of {len(meta_train)} images processed!')


# Save the data for easy loading in the future
torch.save(data, 'images_'+str(img_res)+'.pt')