import os
import numpy as np
from PIL import Image

# Load reshaped_data.npy from the current directory (assuming you are in the correct directory)
reshaped_data = np.load('reshaped_data.npy')
for i in range(100):
	print(reshaped_data[i])
