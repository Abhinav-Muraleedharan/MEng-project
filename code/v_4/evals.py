import torch
import numpy as np
from model import Neuro_behaviour_model
import torch.nn as nn

# Load evaluation data (modify this according to your data preparation)
# For example, if your evaluation data is in 'X_eval.npy' and 'Y_eval.npy':
X_eval = np.load('X_eval.npy')
Y_eval = np.load('Y_eval.npy')

# Preprocessing if necessary (replace NaNs, scaling, etc.)
# Assuming the same preprocessing as in training
X_eval[np.isnan(X_eval)] = 1
Y_eval[np.isnan(Y_eval)] = 0

# Convert data to PyTorch tensors
X_eval_tensor = torch.tensor(X_eval, dtype=torch.float32)
Y_eval_tensor = torch.tensor(Y_eval, dtype=torch.float32)

# Load model (ensure the architecture is the same as the one used for training)
input_size = X_eval.shape[1]
output_size = Y_eval.shape[1]
hidden_layer_sizes = [130, 100, 130, 50, 20, 10, 5, 3]  # Same as used during training
dropout_prob = 0.5

model = Neuro_behaviour_model(input_size, hidden_layer_sizes, output_size, dropout_prob)
model.load_state_dict(torch.load('trained_model_final.pth'))
model.eval()

# Evaluation
criterion = nn.MSELoss()
total_loss = 0
with torch.no_grad():
    for i in range(X_eval.shape[0]):
        X_i = X_eval_tensor[i,:].view(1, -1)
        Y_i = Y_eval_tensor[i,:].view(1, -1)
        outputs = model(X_i)
        loss = criterion(outputs, Y_i)
        total_loss += loss.item()

# Compute average loss
average_loss = total_loss / X_eval.shape[0]
print(f'Average Loss: {average_loss:.4f}')

# Add other metrics if necessary
