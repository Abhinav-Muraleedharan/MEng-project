from data import Retention_data_loader
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchvision 
import numpy as np 

def show_images(images, n_rows=1):
    # Convert the tensor to numpy for visualization
    images = torchvision.utils.make_grid(images).numpy()
    plt.figure(figsize=(15, 15))

    # Transpose the images from (C, H, W) to (H, W, C) and clip values to display properly
    plt.imshow(np.transpose(images, (1, 2, 0)))
    plt.axis('off')
    plt.show()





folder_1 = '/Users/abhinavmuraleedharan/MEng_project/MEng-project/code/v_3/data/raw_data/binary_image_data'
folder_2 = '/Users/abhinavmuraleedharan/MEng_project/MEng-project/code/v_3/data/raw_data/binary_image_data'

ds = Retention_data_loader(folder_1,folder_2)

train_data_loader = DataLoader(ds,batch_size=128,shuffle=True)

images = next(iter(train_data_loader))


# Display images from the first batch
show_images(images, n_rows=3)