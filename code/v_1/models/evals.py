##### Autoregressive generation code #####
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
from u_net import UNet 

# Function to load the trained model
def load_model(model_path):
    model = UNet(in_channels=1, out_channels=1)  # Make sure to replace UNet() with the actual definition of your U-Net model
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint)
    model.eval()
    return model

# Function to perform inference on a new image
def inference(model, image_path,threshold):
    # Load and preprocess the image
    image = Image.open(image_path).convert('L')  # 'L' mode is for grayscale
    transform = transforms.Compose([
        transforms.Resize((64, 64)),  # Resize the image to (256, 256)
        transforms.ToTensor()  # Convert the image to a PyTorch tensor
    ])
    image = transform(image)
    input_image = image.view(1,1,64,64) # Add batch dimension

    # Run inference
    with torch.no_grad():
        output = model(input_image)
    print(output)
    # Process the output as needed
    # For example, convert the output tensor to a numpy array
    print("Dimensions of output image",output.shape)
    output = output.squeeze()
    output = torch.sigmoid(output)
    output = (output * 255).type(torch.uint8)  # Scale to 0-255 and convert to uint8
    output = torch.where(output >= threshold, torch.tensor(255, dtype=torch.uint8), torch.tensor(0, dtype=torch.uint8))

    # Convert to numpy array
    output_np = output.cpu().numpy()
    # Convert to PIL Image
    # The 'L' mode is for grayscale images
    image = Image.fromarray(output_np, 'L')
    # Save or display the image
    image.save('output_image.png')
    image.show()

    return output_np

if __name__ == "__main__":
    # Replace 'path/to/checkpoint.pth' with the actual path to your checkpoint file
    model_path = 'checkpoint_epoch4.pth'
    model = load_model(model_path)

    # Replace 'path/to/new/image.jpg' with the path to your new image
    image_path = '/Users/abhinavmuraleedharan/MEng_project/MEng-project/code/v_1/data/raw_data/binary_image_data/image_3.png'
    result = inference(model, image_path,threshold= 165)

    # Perform further processing or visualization as needed
    print(result)

