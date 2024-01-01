import torch
import torch.nn as nn
from u_net import UNet


class ExtendedUNet(nn.Module):
    def __init__(self, u_net_model, fc_layers, output_size):
        super(ExtendedUNet, self).__init__()
        self.u_net = u_net_model
        
        # Create a Sequential container for the fully connected layers
        fc_layers_list = []
        for in_features, out_features in zip(fc_layers, fc_layers[1:]):
            fc_layers_list.append(nn.Linear(in_features, out_features))
            fc_layers_list.append(nn.ReLU(inplace=True))
        
        self.fc_layers = nn.Sequential(*fc_layers_list)
        
        # Final fully connected layer
        self.final_fc = nn.Linear(fc_layers[-1], output_size)

    def forward(self, x):
        # Forward through U-Net
        x = self.u_net(x)

        # Flatten the output
        x = x.view(x.size(0), -1)

        # Forward through the fully connected layers
        x = self.fc_layers(x)

        # Final layer
        x = self.final_fc(x)

        return x

# Assuming you have loaded your U-Net model as u_net_model
# Define the sizes of the fully connected layers
# fc_layers = [64*64, 512, 128]  # Example sizes, adjust as needed
# output_size = 3  



# Example output size, adjust as needed


# input_size = 100
# hidden_layer_sizes = [10,12,24,25,23,25]
# output_size = 3 
# dropout_prob = 0.5 
# # Create an instance of the custom neural network
# model = Neuro_behaviour_model(input_size, hidden_layer_sizes, output_size, dropout_prob)

# # Print the model architecture
# print(model)
