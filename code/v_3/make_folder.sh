#!/bin/bash

# Create main project directory
mkdir architecture_1
cd architecture_1

# Create subdirectories
mkdir data
mkdir code
mkdir models
mkdir notebooks
mkdir docs

# Inside the 'code' directory, create subdirectories for specific tasks
cd code
mkdir data_processing
mkdir feature_engineering
mkdir model_training
mkdir model_evaluation

# Go back to the main project directory
cd ..

# Create additional directories as needed
# Example:
# mkdir results
# mkdir logs

echo "Folder structure created successfully."

