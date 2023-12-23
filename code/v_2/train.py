# training code with online computation of auxillary variables.
# batch size of 1

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os 
from model import Neuro_behaviour_model


#cuda
# Check for GPU #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
########

# Get the current directory
current_directory = os.getcwd()

# Append the filename to the current directory
x_neural_path = os.path.join(current_directory, 'X_neural.npy')
y_target_path = os.path.join(current_directory, 'Y_target.npy')

x_neural_path =  '/Users/abhinavmuraleedharan/MEng_project/MEng-project/code/v_2/X_neural.npy'
y_target_path =  '/Users/abhinavmuraleedharan/MEng_project/MEng-project/code/v_2/Y_target.npy'
# Load data from npy files
X_neural = np.load(x_neural_path)
Y_target = np.load(y_target_path)

# replace all nan values by 1 in neural spike observations.
X_neural[np.isnan(X_neural)] = 1
# replace all nan values by 0 in observation data.
Y_target[np.isnan(Y_target)] = 0
print("Checking nan in dataset")
print(np.max(X_neural))


# Convert data to PyTorch tensors
X_neural_tensor = torch.tensor(X_neural, dtype=torch.float32).to(device)
Y_target_tensor = torch.tensor(Y_target, dtype=torch.float32).to(device)

contains_nan = torch.any(torch.isnan(X_neural_tensor))
contains_nan = torch.any(torch.isnan(Y_target_tensor))
print(Y_target_tensor[163000])
if contains_nan:
    print("The tensor contains NaN values.")
else:
    print("The tensor does not contain NaN values.")

# Instantiate the model
input_size = X_neural.shape[1]  # Assuming the number of features in X_neural is the input size
output_size = Y_target.shape[1]  # Assuming the number of features in Y_target is the output size
# hidden_layer_sizes = [130, 100,130,100,60,70,50,40,30,20,10,5,3]  # Example hidden layer sizes
hidden_layer_sizes = [130, 100,130,50,20,10,5,3]
dropout_prob = 0.5  #dropout probability

# instantiate model
model = Neuro_behaviour_model(input_size, hidden_layer_sizes, output_size, dropout_prob).to(device)

#print number of parameters::

num_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters in the model: {num_params}")

# Define lodd function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training loop
num_epochs = 1000
print(X_neural.shape[0])

loss_values= []
for epoch in range(num_epochs):
    X_i = torch.zeros(1, input_size, device=device) 
    for i in range(X_neural.shape[0]):
        X_i = 0.5*X_neural_tensor[i,:] + 0.5*X_i 
        # Forward pass
        # if i > 162000:
        #     print(X_i)
        # print(X_i)
        outputs = model(X_i)
        # Compute the loss
        Y = Y_target_tensor[i,:].view(1,3)
        loss = criterion(outputs, Y)
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+ 1) % 10000 == 0:
            print(f'i [{i+1}/{X_neural.shape[0]}], Loss: {loss.item():.4f}')
        # Print the loss every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        loss_values.append(loss.item())
        torch.save(model.state_dict(), 'trained_model.pth')
         # print(X_i)
# After training, save the loss values to a file
with open('training_log.txt', 'w') as file:
    for loss in loss_values:
        file.write(f'{loss:.4f}\n')
# Save the final trained model
torch.save(model.state_dict(), 'trained_model_final.pth')
