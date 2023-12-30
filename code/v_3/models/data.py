import os
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms
from PIL import Image 

class Retention_data_loader(Dataset):
    def __init__(self, folder_1, folder_2, transform = transforms.ToTensor()):
        self.folder_1 = folder_1
        self.folder_2 = folder_2
        self.transform = transform 
         # load file names:
        self.folder1_files = [os.path.join(folder_1, f) for f in os.listdir(folder_1)]
        self.folder2_files =  [os.path.join(folder_2, f) for f in os.listdir(folder_2)]

    def __len__(self):
        return(len(self.folder1_files ))
    
    def __getitem__(self,index):
        print("Index:",index)
        for i in range(index):
            #load image_1
            img_path = os.path.join(self.folder_1,self.folder1_files[i])
            image = Image.open(img_path).convert('L')
            if i == 0:
                image_tensor = 0.5*self.transform(image)
            else:
                image_tensor = 0.5*image_tensor + 0.5*self.transform(image)

        return image_tensor
        



       


