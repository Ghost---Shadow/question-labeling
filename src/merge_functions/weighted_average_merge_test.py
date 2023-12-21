import unittest
import torch
from merge_functions.weighted_average_merge import WeightedAverageMerger
import torch.nn as nn
import torch.optim as optim


# python -m unittest merge_functions.weighted_average_merge_test.TestWeightedAverageMerger -v
class TestWeightedAverageMerger(unittest.TestCase):
    # python -m unittest merge_functions.weighted_average_merge_test.TestWeightedAverageMerger.test_forward -v
    def test_forward(self):
        config = {"architecture": {"semantic_search_model": {"device": "cpu"}}}

        # Initialize the module
        model = WeightedAverageMerger(config)

        # Create example input tensors
        batch_size = 2  # Small batch size for testing
        a = torch.tensor([[1.0] * 768] * batch_size)
        b = torch.tensor([[2.0] * 768] * batch_size)

        # Run the module
        output = model(a, b)

        # Check the shape of the output
        self.assertEqual(output.shape, (batch_size, 768))

        # Check if the output is computed correctly
        # Since the initial value of lambda is random, we can't check for exact values
        # Instead, we check if the output lies between a and b
        self.assertTrue(torch.all(output >= a))
        self.assertTrue(torch.all(output <= b))

    # python -m unittest merge_functions.weighted_average_merge_test.TestWeightedAverageMerger.test_overfit -v
    def test_overfit(self):
        # Configuration and model setup
        config = {"architecture": {"semantic_search_model": {"device": "cpu"}}}
        model = WeightedAverageMerger(config)

        # Small dataset (single batch)
        a = torch.randn(10, 768)
        b = torch.randn(10, 768)
        lambda_true = torch.rand(1)
        targets = (a + lambda_true * b) / (1 + lambda_true)  # Target weighted averages

        # Loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        # Training loop
        for _ in range(100):
            optimizer.zero_grad()
            outputs = model(a, b)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            print(loss)

        # Check if the model has overfitted
        final_loss = loss.item()
        self.assertLess(final_loss, 1e-4)  # Threshold for loss


if __name__ == "__main__":
    unittest.main()
