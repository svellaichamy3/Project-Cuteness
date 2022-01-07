# Import the required libraries
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import time
import copy


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


### Load the Data and Dataloaders

# Images scaled to 224x224 pixels
data = torch.load('img_multiaugment_224.pt')

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


### Initialize the Model

# Takes in a pretrained model passed through additional FC layers and concatenates this to feature embeddings
# of the metadata passed through FC layers

class PawModel(nn.Module):
    def __init__(self, pretrained_model):
        super(PawModel, self).__init__()
        
        # Call the pretrained model and add FC layer
        self.pretrained_model = pretrained_model
        self.added_layers = nn.Linear(1000, 1024)
        
        # Pass the metadata through FC layers
        self.meta_layers = nn.Sequential(nn.Linear(12, 128),
                                         nn.Linear(128, 256),
                                         nn.Linear(256, 512),
                                         nn.Linear(512, 1024))
        
        # Pass the concatenated embeddings through additional FC layers
        self.final_layers = nn.Sequential(nn.Linear(2048, 1024),
                                          nn.Dropout(0.5),
                                          nn.Linear(1024, 512),
                                          nn.Dropout(0.5),
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


# Initialize our pretrained model (EfficientNet)
effnet = models.efficientnet_b7(pretrained=True)

param_idx = 0
# Freeze the first parameters of the pretrained model
for param in effnet.parameters():
    param_idx += 1
    if param_idx <= 600:
        param.requires_grad = False


# Create the new model
paw_model = PawModel(pretrained_model=effnet)
paw_model = paw_model.to(device)

# Get list of parameters to update
params_to_update = []
print("Parameters to learn:")
for name, param in paw_model.named_parameters():
    if param.requires_grad == True:
        params_to_update.append(param)
        print("\t",name)


### Train the Model
def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=10):
    # Start the timer
    timer = time.time()
    
    # Initialize list for storing validation loss
    val_loss_history = []

    # Define the best model parameters and best loss
    best_params = copy.deepcopy(model.state_dict())
    best_loss = np.inf

    # Iterate through each epoch
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1} of {num_epochs}:')
        print('=' * 15)

        # Alternate between training and validation
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0

            # Iterate over the data
            for (inputs, meta, labels) in dataloaders[phase]:
                inputs = inputs.to(device)
                meta = meta.float().to(device) 
                labels = labels.float().to(device)

                # Reset/zero out the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                # Track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs, meta)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # Backward pass and optimize if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Store statistics
                running_loss += loss.item() * inputs.size(0)
            
            
            epoch_loss = running_loss / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f}'.format(phase, epoch_loss))

            # Copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_params = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_loss_history.append(epoch_loss)
                scheduler.step(epoch_loss)

        print()

    total_time = time.time() - timer
    print('Training complete in {:.0f}m {:.0f}s'.format(total_time // 60, total_time % 60))

    # Load and return the best model
    model.load_state_dict(best_params)
    return model, val_loss_history

criterion = nn.MSELoss()
optimizer = optim.AdamW(params_to_update, lr=0.0001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=1)

trained_model, val_losses = train_model(paw_model, dataloaders_dict,
                                        criterion, optimizer, scheduler, num_epochs=10)


# Save the trained model
torch.save(trained_model, 'best_effnet.pth')