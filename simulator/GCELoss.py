"""
GCELoss.py

=== SUMMARY ===
Description     : Class to implement generalized cross entropy
Date Created    : July 26, 2020
Last Updated    : July 26, 2020

=== DETAILED DESCRIPTION ===
 > see resources/generalized_cross_entropy.txt for more details
 > NOTE: this implementation does not include the size_average or weight parameters found
   in other PyTorch loss functions

=== UPDATE NOTES ===
 > July 26, 2020
    - file created
"""

import torch
import torch.nn as nn


class GCELoss(nn.Module):
    def __init__(self, reduction='none') -> None:
        super(GCELoss, self).__init__()
        assert reduction in ['none', 'mean', 'sum'], "ERROR: Invalid reduction method specified."
        self.reduction = reduction

    def forward(self, output, target):
        gce_loss = torch.mul(target, torch.log(torch.div(target, output))) + \
            torch.mul((1 - target), torch.log(torch.div(1 - target, 1 - output)))

        if self.reduction == 'none':
            return gce_loss
        elif self.reduction == 'sum':
            return gce_loss.sum()
        else:
            return gce_loss.mean()

