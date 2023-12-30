import os
import numpy as np
from PIL import Image

# Load reshaped_data.npy from the current directory (assuming you are in the correct directory)
reshaped_data = np.load('reshaped_data.npy')

# Set the threshold value
threshold = 0.1

# Apply thresholding: set pixels > threshold to 255, and <= threshold to 0
thresholded_data = np.where(reshaped_data > threshold, 255, 0)

# Create a folder to save images
output_folder = 'binary_image_data'
os.makedirs(output_folder, exist_ok=True)

# Save each row of the thresholded data array as an image
for i, image_array in enumerate(thresholded_data):
    # Create an Image object from the thresholded NumPy array
    image = Image.fromarray(image_array.astype(np.uint8))

    # Save the image with filename as the row index
    image_filename = os.path.join(output_folder, f'image_{i}.png')
    image.save(image_filename)

print("Thresholded images saved successfully.")

