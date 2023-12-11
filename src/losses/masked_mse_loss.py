import torch.nn as nn


class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, input, target, mask):
        # Ensure that the mask is a boolean tensor
        mask = mask.bool()

        # Apply the mask
        masked_input = input[~mask]
        masked_target = target[~mask]

        # Compute the MSE loss
        return nn.functional.mse_loss(masked_input, masked_target, reduction="mean")
