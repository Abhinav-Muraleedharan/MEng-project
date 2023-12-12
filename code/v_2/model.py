import torch
import torch.nn as nn

class Neuro_behaviour_model(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_prob=0.0):
        super(Neuro_behaviour_model
    , self).__init__()

        # Create a list to hold the layers
        layers_list = []

        # Add the input layer
        layers_list.append(nn.Linear(input_size, hidden_sizes[0]))
        layers_list.append(nn.ReLU())
        layers_list.append(nn.Dropout(p=dropout_prob))

        # Add the hidden layers with dropout
        for i in range(1, len(hidden_sizes)):
            layers_list.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))
            layers_list.append(nn.BatchNorm1d(hidden_sizes[i]))
            layers_list.append(nn.ReLU())
            layers_list.append(nn.Dropout(p=dropout_prob))

        # Add the output layer
        layers_list.append(nn.Linear(hidden_sizes[-1], output_size))

        # Create a sequential container for the layers
        self.layers = nn.Sequential(*layers_list)

    def forward(self, x):
        return self.layers(x)

# input_size = 100
# hidden_layer_sizes = [10,12,24,25,23,25]
# output_size = 3 
# dropout_prob = 0.5 
# # Create an instance of the custom neural network
# model = Neuro_behaviour_model(input_size, hidden_layer_sizes, output_size, dropout_prob)

# # Print the model architecture
# print(model)
