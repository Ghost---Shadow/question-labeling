import torch
import torch.nn as nn
from pydash import get


class DnnMerger(nn.Module):
    def __init__(self, config):
        super(DnnMerger, self).__init__()

        num_layers = config["architecture"]["aggregation_strategy"]["merge_strategy"][
            "num_layers"
        ]
        device = config["architecture"]["semantic_search_model"]["device"]

        # Computed field
        input_size = config["architecture"]["semantic_search_model"]["output_dim"]

        output_size = input_size
        hidden_size = input_size * 2

        layers = []

        # Adjust input size to account for concatenated inputs
        adjusted_input_size = input_size * 2

        # Input layer
        layers.append(nn.Linear(adjusted_input_size, hidden_size, device=device))
        layers.append(nn.ReLU())

        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_size, hidden_size, device=device))
            layers.append(nn.ReLU())

        # Output layer
        layers.append(nn.Linear(hidden_size, output_size, device=device))

        # Combine all layers
        self.layers = nn.Sequential(*layers)

    def forward(self, a, b):
        combined_input = torch.cat((a, b), dim=1)
        return self.layers(combined_input)

    def get_metrics(self):
        metrics = {}
        return metrics
