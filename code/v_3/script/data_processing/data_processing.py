import os
import numpy as np
from PIL import Image

# Load reshaped_data.npy from the current directory (assuming you are in the correct directory)
reshaped_data = np.load('reshaped_data.npy')

# Create a folder to save images
output_folder = 'image_data'
os.makedirs(output_folder, exist_ok=True)

# Save each row of the reshaped data array as an image
for i, image_array in enumerate(reshaped_data):
    # Create an Image object from the NumPy array
    image = Image.fromarray((image_array * 255).astype(np.uint8))  # Assuming pixel values are in [0, 1] range

    # Save the image with filename as the row index
    image_filename = os.path.join(output_folder, f'image_{i}.png')
    image.save(image_filename)

print("Images saved successfully.")

