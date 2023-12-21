import torch
import torch.nn as nn
from pydash import get


class DnnMerger(nn.Module):
    def __init__(
        self,
        input_size,
        config,
    ):
        super(DnnMerger, self).__init__()

        output_size = input_size
        hidden_size = input_size * 2

        num_layers = get(
            config, "architecture.aggregation_strategy.merge_strategy.num_layers"
        )
        layers = []

        # Adjust input size to account for concatenated inputs
        adjusted_input_size = input_size * 2

        # Input layer
        layers.append(nn.Linear(adjusted_input_size, hidden_size))
        layers.append(nn.ReLU())

        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())

        # Output layer
        layers.append(nn.Linear(hidden_size, output_size))

        # Combine all layers
        self.layers = nn.Sequential(*layers)

    def forward(self, a, b):
        combined_input = torch.cat((a, b), dim=1)
        return self.layers(combined_input)

    def get_metrics(self):
        metrics = {}
        return metrics
