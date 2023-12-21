import unittest
import torch
from merge_functions.dnn_merge import DnnMerger
import torch.nn as nn
import torch.optim as optim


# python -m unittest merge_functions.dnn_merge_test.TestDnnMerger -v
class TestDnnMerger(unittest.TestCase):
    # python -m unittest merge_functions.dnn_merge_test.TestDnnMerger.test_forward -v
    def test_forward(self):
        # Configuration for the network
        config = {
            "architecture": {
                "aggregation_strategy": {"merge_strategy": {"num_layers": 4}}
            }
        }

        # Initialize the model
        input_size = 768  # Example input size for each of a and b
        model = DnnMerger(input_size, config)

        # Create example input tensors
        a = torch.randn(1, input_size)
        b = torch.randn(1, input_size)

        # Run the model
        output = model(a, b)

        # Check the shape of the output
        expected_output_shape = (1, input_size)
        self.assertEqual(output.shape, expected_output_shape)

    # python -m unittest merge_functions.dnn_merge_test.TestDnnMerger.test_overfit -v
    def test_overfit(self):
        # Configuration and model setup
        input_size = 768
        config = {
            "architecture": {
                "aggregation_strategy": {"merge_strategy": {"num_layers": 4}}
            }
        }
        model = DnnMerger(input_size, config)

        # Small dataset (single batch)
        a = torch.randn(10, input_size)
        b = torch.randn(10, input_size)
        targets = torch.randn(10, input_size)  # Random targets

        # Loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Training loop
        for _ in range(100):  # Number of iterations
            optimizer.zero_grad()
            outputs = model(a, b)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            print(loss)

        # Check if the model has overfitted
        final_loss = loss.item()
        self.assertLess(final_loss, 0.1)  # Threshold for loss


if __name__ == "__main__":
    unittest.main()
