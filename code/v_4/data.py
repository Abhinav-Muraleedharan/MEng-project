import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np


class CustomDataset(Dataset):
    def __init__(self, img_dir, npy_file, transform):
        """
        Args:
            img_dir (string): Directory with all the images.
            npy_file (string): Path to the .npy file containing the numpy matrix.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.img_dir = img_dir
        self.npy_data = np.load(npy_file)
        self.transform = transform

        # Ensure we have the same number of images and numpy data points
        self.img_names = sorted(os.listdir(img_dir))
        # assert len(self.img_names) == len(self.npy_data), \
        #     "Mismatch between image and numpy data lengths"

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = f'image_{idx}.png'
        img_loc = os.path.join(self.img_dir, img_name)
        image = Image.open(img_loc)

        if self.transform:
            image = self.transform(image)

        numpy_data = torch.tensor(self.npy_data[idx], dtype=torch.float32)

        sample = {'image': image, 'numpy_data': numpy_data}

        return sample
