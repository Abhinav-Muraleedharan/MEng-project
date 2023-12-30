##### Autoregressive generation code #####
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
from u_net import UNet 
from torchviz import make_dot


# Function to load the trained model
def load_model(model_path):
    model = UNet(n_channels=1, n_classes=1)  # Make sure to replace UNet() with the actual definition of your U-Net model
    # checkpoint = torch.load(model_path)
    # model.load_state_dict(checkpoint)
    # model.eval()
    return model

# Function to perform inference on a new image
def inference(model, image_path,threshold):
    # Load and preprocess the image
    image = Image.open(image_path).convert('L')  # 'L' mode is for grayscale
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize the image to (256, 256)
        transforms.ToTensor()  # Convert the image to a PyTorch tensor
    ])
    image = transform(image)
    input_image = image.view(1,1,256,256) # Add batch dimension

    # Run inference
    
    output = model(input_image)
    print("Shape of output", output.shape)
    # Visualize the graph
    dot = make_dot(output, params=dict(list(model.named_parameters())))
    dot.render('model_graph', format='png')  # Saves the graph as a PNG image
    #### save netron file #####
    torch.onnx.export(model, input_image, "U_Net_model.onnx",opset_version = 11)



if __name__ == "__main__":
    # Replace 'path/to/checkpoint.pth' with the actual path to your checkpoint file
    model_path = 'checkpoint_epoch1.pth'
    model = load_model(model_path)

    # Replace 'path/to/new/image.jpg' with the path to your new image
    image_path = '/Users/abhinavmuraleedharan/MEng_project/MEng-project/code/v_1/data/raw_data/binary_image_data/image_3.png'
    result = inference(model, image_path,threshold= 210)

    # Perform further processing or visualization as needed
    print(result)

