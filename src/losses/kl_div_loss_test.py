import unittest
import torch
import torch.nn.functional as F
from losses.kl_div_loss import KLDivLoss


class TestKLDivLoss(unittest.TestCase):
    # python -m unittest losses.kl_div_loss_test.TestKLDivLoss.test_loss_computation -v
    def test_loss_computation(self):
        input_tensor = torch.tensor(
            [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], requires_grad=True
        )
        target_tensor = torch.tensor([[1.1, 2.2, 3.3], [1.1, 2.2, 3.3]])

        # Convert to probability distributions
        input_prob = F.softmax(input_tensor, dim=-1)
        target_prob = F.softmax(target_tensor, dim=-1)

        expected_loss = torch.tensor([0.00021061301231384277])

        loss_fn = KLDivLoss()

        loss = loss_fn(input_prob, target_prob)

        self.assertAlmostEqual(loss.item(), expected_loss.item(), places=4)

        # Should not crash
        loss.backward()


if __name__ == "__main__":
    unittest.main()
