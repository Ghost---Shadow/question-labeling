import torch


class KneeLoss(torch.nn.Module):
    def forward(self, similarities, num_relevant):
        # Set first derivative to be the similarities themselves
        first_derivative = similarities

        # Compute second derivative
        second_derivative = torch.diff(
            similarities, append=torch.tensor([0.0], device=similarities.device)
        )

        # Loss for non-distractors
        non_distractor_loss = torch.sum(
            torch.relu(-first_derivative[:num_relevant])
        ) + torch.sum(torch.relu(second_derivative[:num_relevant]))

        # Loss for distractors
        distractor_loss = torch.sum(
            torch.square(first_derivative[num_relevant:])
        ) + torch.sum(torch.square(second_derivative[num_relevant:]))

        # Total loss
        total_loss = non_distractor_loss + distractor_loss
        return total_loss
