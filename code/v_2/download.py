import subprocess

# Install required packages
subprocess.run(['pip', 'install', 'git+https://github.com/neurallatents/nlb_tools.git'])
subprocess.run(['pip', 'install', 'dandi'])

# Download dataset
subprocess.run(['dandi', 'download', 'https://gui.dandiarchive.org/#/dandiset/000127'])
