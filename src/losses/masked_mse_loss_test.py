import unittest
import torch
from losses.masked_mse_loss import MaskedMSELoss


# python -m unittest losses.masked_mse_loss_test.TestMaskedMSELoss -v
class TestMaskedMSELoss(unittest.TestCase):
    # python -m unittest losses.masked_mse_loss_test.TestMaskedMSELoss.test_loss_computation -v
    def test_loss_computation(self):
        input_tensor = torch.tensor(
            [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], requires_grad=True
        )
        target_tensor = torch.tensor([[1.1, 2.2, 3.3], [1.1, 2.2, 3.3]])
        mask_tensor = torch.tensor([[False, True, False], [False, False, True]])

        expected_loss = torch.tensor([0.03750000149011612])
        loss_fn = MaskedMSELoss()

        loss = loss_fn(input_tensor, target_tensor, mask_tensor)

        self.assertAlmostEqual(loss.item(), expected_loss.item(), places=4)

        # Should not crash
        loss.backward()


if __name__ == "__main__":
    unittest.main()
