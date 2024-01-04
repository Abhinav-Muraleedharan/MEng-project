## code for generating samples of neural activity:
from model import UNet
from torch.vision import transforms 

#write function that accepts retention input and generates an output image

#

def generate(img_input, model):
    ##convert image to pytorch tensor
    x_tensor = transform(img_input)
    output = model(x_tensor)

    # based on threshold, generate an output image

    



# load model from drive #




## load initial image ###

### generate one image

