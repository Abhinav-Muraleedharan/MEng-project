# training code with online computation of auxillary variables.
# batch size of 1
from u_net import UNet 
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os 
from model import ExtendedUNet
from torchvision import transforms
from data import CustomDataset
from torch.utils.data import DataLoader
import json
#cuda
# Check for GPU #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
########

#load UNet model 

u_net_model = UNet(n_channels=1,n_classes=1)
u_net_model.load_state_dict(torch.load('checkpoint_epoch1.pth'))
fc_layers = [64*64, 512,512,512,256, 128,64,32,16,8]  # Example sizes, adjust as needed
output_size = 3  ###

# instantiate model
model = ExtendedUNet(u_net_model, fc_layers, output_size).to(device)

#print number of parameters::

num_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters in the model: {num_params}")

# Define lodd function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# Create the dataset
dataset = CustomDataset(img_dir='image_data/', 
                        npy_file='Y_target.npy', 
                        transform=transform)

# Create the DataLoader
data_loader = DataLoader(dataset, batch_size=128, shuffle=True)
model = model.float()
# Number of epochs
num_epochs = 10  # You can modify this number based on your requirements
print("Length of dataloader",len(data_loader))
# Transfer model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
train_losses_1 = []
train_losses_2 = []
# Training Loop
i = 0 
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    i = 0
    for batch in data_loader:
        print("in training loop")
        # Get data
        images = batch['image'].to(device)
        numpy_data = batch['numpy_data'].to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        
        # Compute the loss
        loss = criterion(outputs, numpy_data)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        print(loss.item())
        train_losses_1.append({'epoch': epoch, 'i': i, 'training_loss': loss.item()})
        if i%100 == 0:
            print("Saving Checkpoint:")
            checkpoint_path = f'checkpoint_epoch{epoch + 1}.pth'
            torch.save(model.state_dict(), checkpoint_path)
            losses_str = json.dumps(train_losses_1, indent=4)
            with open('training_log_1.txt', 'w') as file:
                file.write(losses_str)
        running_loss += loss.item()
        i = i + 1 

    # Print statistics
    epoch_loss = running_loss / len(data_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
    # append training loss 
    train_losses_2.append({'epoch': epoch, 'i': i, 'epoch_loss': epoch_loss})
    checkpoint_path = f'checkpoint_epoch{epoch + 1}.pth' # checkpoint path
    torch.save(model.state_dict(), checkpoint_path) # save model 
    losses_str = json.dumps(train_losses_2, indent=4)
    with open('training_log_2.txt', 'w') as file:
        file.write(losses_str)

print("Training complete")

# Save the final trained model
torch.save(model.state_dict(), 'trained_model_final.pth')
