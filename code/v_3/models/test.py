#test code
import os
import torch
import torchvision.transforms.functional
from torch import nn
from u_net import UNet
from PIL import Image
from torchvision import transforms

data_folder = '/Users/abhinavmuraleedharan/meng_proj/v_1/data/raw_data/binary_image_data'

# Define a custom function to load images based on filenames
def load_images(file_prefix, num_images):
    images = []
    for i in range(1, num_images + 1):
        image_filename = os.path.join(data_folder, f'{file_prefix}_{i}.png')
        if os.path.exists(image_filename):
            # Open the image and convert it to grayscale and then to a PyTorch tensor
            image = Image.open(image_filename).convert('L')  # 'L' mode is for grayscale
            transform = transforms.Compose([
                transforms.Resize((64, 64)),  # Resize the image to (256, 256)
                transforms.ToTensor()  # Convert the image to a PyTorch tensor
            ])
            image = transform(image)
            images.append(image)
        else:
            print(f"Image file '{image_filename}' not found.")
    return images
    
# Example usage
file_prefix = 'image'
num_images = 10  # Number of images with filenames like 'image_1.png', 'image_2.png', ..., 'image_10.png'
loaded_images = load_images(file_prefix, num_images)
x = loaded_images[0].view(1,1,64,64)
print(x.size())
print(x)
in_channels = 1
out_channels = 1
model = UNet(in_channels,out_channels)


y = model.forward(x)
print(y)
print(" y size:",y.size())


