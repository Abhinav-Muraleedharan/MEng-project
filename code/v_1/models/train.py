import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from u_net import UNet
from PIL import Image
from torchvision import transforms

# Define U-Net architecture (same as before)

# Hyperparameters
learning_rate = 0.001
batch_size = 1
num_epochs = 10
# data folder:

data_folder = '/Users/abhinavmuraleedharan/meng_proj/v_1/data/raw_data/binary_image_data'
# Create U-Net model, loss function, and optimizer
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("mps")
model = UNet(in_channels=1, out_channels=1).to(device)
criterion = nn.BCEWithLogitsLoss()  # Binary Cross-Entropy loss for binary segmentation
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# function for loading i th image:

def load_image(i):
    image_filename = os.path.join(data_folder, f'image_{i}.png')
    if os.path.exists(image_filename):
    # Open the image and convert it to grayscale and then to a PyTorch tensor
        image = Image.open(image_filename).convert('L')  # 'L' mode is for grayscale
        transform = transforms.Compose([
        transforms.Resize((64, 64)),  # Resize the image to (256, 256)
        transforms.ToTensor()  # Convert the image to a PyTorch tensor
        ])
        image = transform(image)
    else:
        print("Wrong filepath")
    image = image.view(1,1,64,64)
    return image


batch_idx = 100

train_losses = []
# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    x_input = torch.zeros(1, 1, 64, 64)
    for i in range(28000):
        # get inputs
        x_input = 2**(-1)*load_image(i) + 2**(-1)*x_input
        targets = load_image(i+1)
        inputs, targets = x_input.to(device), targets.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        
        # Calculate the loss
        loss = criterion(outputs, targets)
        
        # Backpropagation and optimization
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        loss_val = loss.item()
        
        # Print statistics every 10 batches
        if i % 100 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Image: {i}, Loss: {running_loss / 100:.4f}")
            train_losses.append(running_loss/100)
            running_loss = 0.0
                    
        # save model when at every 2000 th i
        if i%2000 == 0:
            print("Saving Checkpoint:")
            checkpoint_path = f'checkpoint_epoch{epoch + 1}.pth'
            torch.save(model.state_dict(), checkpoint_path)
            with open('training_log.txt', 'a') as file:
                for loss in train_losses:
                    file.write(f'{loss}\n')
            
       

print("Training finished!")

