import unittest
import torch
import torch.nn.functional as F
from losses.knee_loss import KneeLoss
from tqdm import tqdm
import matplotlib.pyplot as plt


class TestKneeLoss(unittest.TestCase):
    # python -m unittest losses.knee_loss_test.TestKneeLoss.test_loss_computation -v
    def test_loss_computation(self):
        num_relevant = 10
        num_distractor = 20

        similarities = torch.rand(
            num_relevant + num_distractor, device="cpu", requires_grad=True
        )

        loss_fn = KneeLoss({})

        loss = loss_fn(similarities, num_relevant)

        # Should not crash
        loss.backward()

    # python -m unittest losses.knee_loss_test.TestKneeLoss.test_minima -v
    def test_minima(self):
        num_relevant = 2

        similarities = torch.tensor(
            [0, 0, 0, 0], requires_grad=True, dtype=torch.float32
        )

        loss_fn = KneeLoss({})

        loss = loss_fn(similarities, num_relevant)
        print(loss)

        # Should not crash
        loss.backward()

    # python -m unittest losses.knee_loss_test.TestKneeLoss.test_training -v
    def test_training(self):
        num_relevant = 10
        num_distractor = 20

        similarities = torch.rand(
            num_relevant + num_distractor, device="cpu", requires_grad=True
        )

        loss_fn = KneeLoss({})

        optimizer = torch.optim.AdamW([similarities], lr=1e-1)

        # Store the cumulative sum progress
        cumsum_progress = []

        num_epochs = 100
        plot_every = 20
        for epoch in tqdm(range(num_epochs)):
            optimizer.zero_grad()
            loss = loss_fn(similarities, num_relevant)
            loss.backward()
            optimizer.step()
            similarities.data = torch.clamp(similarities.data, 0, 1)
            if epoch % plot_every == 0 or epoch == num_epochs - 1:
                cumsum_progress.append(
                    similarities.data.clone().detach().cumsum(0).cpu().numpy()
                )

        # Plotting the cumulative sum evolution
        plt.figure(figsize=(12, 8))
        for i, cumsum in enumerate(cumsum_progress):
            # if i < 1:
            #     continue
            plt.plot(cumsum, marker="o", label=f"Epoch {i * plot_every}")
        ideal_curve = loss_fn._ideal_curve(similarities, num_relevant).cumsum(0).numpy()
        plt.plot(ideal_curve, color="r", linestyle="--", label=f"Ideal")
        plt.axvline(
            x=num_relevant, color="r", linestyle="--", label="Knee (x = num_relevant)"
        )
        plt.title("Evolution of Cumulative Sum of Similarities Over Training (AdamW)")
        plt.xlabel("Document Index")
        plt.ylabel("Cumulative Sum of Similarities")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("./artifacts/knee_loss.png")


if __name__ == "__main__":
    unittest.main()
