import torch.nn as nn


class NoOpModule(nn.Module):
    def __init__(self, *args, **kwargs):
        super(NoOpModule, self).__init__()

    def forward(self, x):
        raise NotImplementedError("Forward method is not implemented in NoOpModule")

    def get_metrics(self):
        return {}

    def parameters(self):
        return []
