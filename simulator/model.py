"""
model.py

=== SUMMARY ===
Description     : Define the model architecture
Date Created    : May 03, 2020
Last Updated    : September 28, 2020

=== DETAILED DESCRIPTION ===
 > Model Architecture
    - 105 input units, 61 output units
    - one hidden layer fo 100 units
    - weights initialized to have uniform distribution of -0.1 to 0.1
    - bias initialized to set value of -1.85
    - activation functions: sigmoid


=== UPDATE NOTES ===
 > September 28, 2020
    - update function docstrings
 > July 18, 2020
    - minor reformatting changes
 > May 24, 2020
    - modify forward pass to return hidden layer activation as well
 > May 03, 2020
    - file created, code copied from Plaut_Model (v1)
"""

import torch
import torch.nn as nn


class PlautNet(nn.Module):
    def __init__(self):
        """
        Initializes model by defining architecture and initializing weights

        Returns:
            None
        """
        super(PlautNet, self).__init__()
        self.layer1 = nn.Linear(105, 100)
        self.layer2 = nn.Linear(100, 61)
        self.init_weights()

    def init_weights(self):
        """
        Initializes weights and bias according to description above

        Returns:
            None
        """
        init_range = 0.1

        self.layer1.weight.data.uniform_(-init_range, init_range)
        self.layer1.bias.data.uniform_(-1.85, -1.85)

        self.layer2.weight.data.uniform_(-init_range, init_range)
        self.layer2.bias.data.uniform_(-1.85, -1.85)

    def forward(self, x):
        """
        Implements forward pass of the model

        Arguments:
            x (torch.Tensor): input to the model

        Returns:
             (torch.Tensor, torch.Tensor): hidden layer activations, output layer activations
        """
        x = torch.sigmoid(self.layer1(x))
        return x, torch.sigmoid(self.layer2(x))
