import torch.nn as nn


class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, input, target):
        # Compute the MSE loss
        return nn.functional.mse_loss(input, target, reduction="mean")
