import unittest
import torch
from aggregation_strategies.weighted_average_strategy import WeightedEmbeddingAverage


# python -m unittest aggregation_strategies.weighted_average_strategy_test.TestWeightedEmbeddingAverage -v
class TestWeightedEmbeddingAverage(unittest.TestCase):
    # python -m unittest aggregation_strategies.weighted_average_strategy_test.TestWeightedEmbeddingAverage.test_weighted_embedding_average -v
    def test_weighted_embedding_average(self):
        # Create sample data
        question_embedding = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        document_embeddings = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], requires_grad=True
        )
        mask = torch.tensor([True, False, True])

        # Create an instance of the WeightedEmbeddingAverage class
        config = {}  # Example configuration, adjust as needed
        model_ref = None  # Example model reference, adjust as needed
        weighted_average_module = WeightedEmbeddingAverage(config, model_ref)

        # Expected result (adjust as needed)
        expected_weighted_average = torch.tensor([0.4016, 0.5623, 0.7229])

        # Run the method
        average_embedding = weighted_average_module(
            question_embedding, document_embeddings, mask
        )

        # Check if the results are as expected
        self.assertTrue(
            torch.allclose(average_embedding, expected_weighted_average, atol=1e-3)
        )

        # Compute the loss
        loss_function = torch.nn.MSELoss()
        loss = loss_function(average_embedding, torch.zeros_like(average_embedding))

        # Backward pass
        loss.backward()

        # Check if gradients are not None (indicating backward pass worked)
        self.assertIsNotNone(question_embedding.grad)
        self.assertIsNotNone(document_embeddings.grad)

    # python -m unittest aggregation_strategies.weighted_average_strategy_test.TestWeightedEmbeddingAverage.test_empty_mask -v
    def test_empty_mask(self):
        # Create sample data
        question_embedding = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        document_embeddings = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], requires_grad=True
        )
        mask = torch.tensor([False, False, False])

        # Create an instance of WeightedEmbeddingAverage
        config = {}  # Example configuration, adjust as needed
        model_ref = None  # Example model reference, adjust as needed
        weighted_average_module = WeightedEmbeddingAverage(config, model_ref)

        # Expected result
        expected_weighted_average = question_embedding

        # Run the method
        average_embedding = weighted_average_module(
            question_embedding, document_embeddings, mask
        )

        # Check if the results are as expected
        self.assertTrue(
            torch.allclose(average_embedding, expected_weighted_average, atol=1e-6)
        )

        # Compute the loss
        loss_function = torch.nn.MSELoss()
        loss = loss_function(average_embedding, torch.zeros_like(average_embedding))

        # Backward pass
        loss.backward()

        # Check if gradients are not None (indicating backward pass worked)
        self.assertIsNotNone(question_embedding.grad)

        # This should be None for an empty mask
        self.assertIsNone(document_embeddings.grad)


if __name__ == "__main__":
    unittest.main()
