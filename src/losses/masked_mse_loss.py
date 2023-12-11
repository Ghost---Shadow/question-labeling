import torch.nn as nn


class MaskedMSELoss(nn.Module):
    def __init__(self, config=None):
        super(MaskedMSELoss, self).__init__()
        self.config = config

    def forward(self, input, target):
        # Compute the MSE loss
        return nn.functional.mse_loss(input, target, reduction="mean")
