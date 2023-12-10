import os
import subprocess

# Install required packages
subprocess.run(['pip', 'install', 'git+https://github.com/neurallatents/nlb_tools.git'])
subprocess.run(['pip', 'install', 'dandi'])

# Get the current working directory
current_directory = os.getcwd()

# Set the target directory within the current working directory
target_directory = os.path.join(current_directory, 'dataset')
os.makedirs(target_directory, exist_ok=True)

# Download dataset
subprocess.run(['dandi', 'download', 'https://gui.dandiarchive.org/#/dandiset/000127', '-o', target_directory])
