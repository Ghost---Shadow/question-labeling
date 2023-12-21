import torch
import torch.nn as nn


class WeightedAverageMerger(nn.Module):
    def __init__(self, config):
        super(WeightedAverageMerger, self).__init__()
        self.lambda_param = nn.Parameter(torch.rand(1))
        self.config = config

    def forward(self, a, b):
        weighted_average = (a + self.lambda_param * b) / (1 + self.lambda_param)
        return weighted_average

    def get_metrics(self):
        metrics = {"lambda": self.lambda_param.item()}
        return metrics
