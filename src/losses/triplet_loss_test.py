import unittest
import torch
from losses.triplet_loss import TripletLoss


# python -m unittest losses.triplet_loss_test.TestTripletLoss -v
class TestTripletLoss(unittest.TestCase):
    def test_loss_computation(self):
        input_tensor = torch.tensor(
            [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], requires_grad=True
        )
        target_tensor = torch.tensor([[1, 0, 1], [0, 1, 0]])

        expected_loss = torch.tensor([1.0])

        config = {}

        loss_fn = TripletLoss(config)

        loss = loss_fn(input_tensor, target_tensor)

        self.assertAlmostEqual(loss.item(), expected_loss.item(), places=4)

        loss.backward()


if __name__ == "__main__":
    unittest.main()
