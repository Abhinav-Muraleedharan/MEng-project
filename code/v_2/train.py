import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os 
from model import Neuro_behaviour_model
# Load data from npy files
X_neural = np.load('X_neural.npy')
Y_target = np.load('Y_target.npy')

# Convert data to PyTorch tensors
X_neural_tensor = torch.tensor(X_neural, dtype=torch.float32)
Y_target_tensor = torch.tensor(Y_target, dtype=torch.float32)

# Instantiate the model
input_size = X_neural.shape[1]  # Assuming the number of features in X_neural is the input size
output_size = Y_target.shape[1]  # Assuming the number of features in Y_target is the output size
hidden_layer_sizes = [64, 32]  # Example hidden layer sizes
dropout_prob = 0.5  #dropout probability

# instantiate model
model = Neuro_behaviour_model(input_size, hidden_layer_sizes, output_size, dropout_prob)

# Define the loss function and the optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 1000

for epoch in range(num_epochs):
    # Forward pass
    
    outputs = model(X_neural_tensor)
    
    # Compute the loss
    loss = criterion(outputs, Y_target_tensor)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print the loss every 100 epochs
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Save the trained model
torch.save(model.state_dict(), 'trained_model.pth')
