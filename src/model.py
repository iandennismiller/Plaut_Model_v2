"""
model.py

=== SUMMARY ===
Description     : Define the model architecture
Date Created    : May 03, 2020
Last Updated    : May 03, 2020

=== DETAILED DESCRIPTION ===
 > Model Architecture
    - 105 input units, 61 output units
    - one hidden layer fo 100 units
    - weights initialized to have uniform distribution of -0.1 to 0.1
    - bias initialized to set value of -1.85
    - activation functions: sigmoid


=== UPDATE NOTES ===
 > May 03, 2020
    - file created, code copied from Plaut_Model (v1)
"""

import torch
import torch.nn as nn

class Plaut_Net(nn.Module):
    def __init__(self):
        """
        Initializes model by defining architecture and initializing weights
        """
        super(Plaut_Net, self).__init__()
        self.layer1 = nn.Linear(105, 100)
        self.layer2 = nn.Linear(100, 61)
        self.init_weights()
        
    def init_weights(self):
        """
        Initializes weights and bias according to description above
        """
        initrange = 0.1

        self.layer1.weight.data.uniform_(-initrange, initrange)
        self.layer1.bias.data.uniform_(-1.85, -1.85)
        
        self.layer2.weight.data.uniform_(-initrange, initrange)
        self.layer2.bias.data.uniform_(-1.85, -1.85)
    
    def forward(self, x):
        """
        Implements forward pass of the model
        """
        x = torch.sigmoid(self.layer1(x))
        return torch.sigmoid(self.layer2(x))
