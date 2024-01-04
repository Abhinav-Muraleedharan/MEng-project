##### Autoregressive generation code #####
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
from u_net import UNet 

# Function to load the trained model
def load_model(model_path):
    model = UNet(n_channels=1, n_classes=1)  # Make sure to replace UNet() with the actual definition of your U-Net model
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)
    model.eval()
    return model

# Function to perform inference on a new image
def inference(model, input_image,threshold):
    # Load and preprocess the image


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
    
    # image.show()

    return image

if __name__ == "__main__":
    # Replace 'path/to/checkpoint.pth' with the actual path to your checkpoint file
    model_path = 'checkpoint_epoch37.pth'
    model = load_model(model_path)
    i = 0 
    #### load initial image ####
    image_path = '/Users/abhinavmuraleedharan/MEng_project/MEng-project/code/v_1/data/raw_data/binary_image_data/image_0.png'
    image = Image.open(image_path).convert('L')  # 'L' mode is for grayscale
    transform = transforms.Compose([
                transforms.Resize((64, 64)),  # Resize the image to (256, 256)
                transforms.ToTensor()  # Convert the image to a PyTorch tensor
                 ])
    image = transform(image)
    x_i = image.view(1,1,64,64) # Add batch dimension
    z_i = torch.zeros(1,1,64,64)
    for i in range(1000):
        z_i = 0.5*x_i + 0.5*z_i
        y_image = inference(model,z_i,threshold = 120)
        filename = f'/Users/abhinavmuraleedharan/MEng_project/MEng-project/code/v_3/generated_images/output_image{i}.png'
        y_image.save(filename)
        z_image = transform(y_image)
        z_i = z_image.view(1,1,64,64)


    # Replace 'path/to/new/image.jpg' with the path to your new image
    
    # result = inference(model, image_path,threshold= 128)


    # Perform further processing or visualization as needed
    # print(result)

