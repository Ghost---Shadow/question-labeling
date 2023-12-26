import torch


class KneeLoss(torch.nn.Module):
    def __init__(self, config=None):
        super(KneeLoss, self).__init__()
        self.config = config
        self.distractor_retardation_factor = 1e1
        self.min_non_distractor_acceleration = 1e-2

    def _calculate_velocity_acceleration(self, displacement, time):
        """
        Calculate the initial velocity and constant acceleration to reach a given displacement in a given time.

        displacement: The total displacement to be achieved.
        time: The total time (number of steps) to achieve the displacement.
        """
        # Assuming final velocity is zero (decelerating to a stop), use kinematic equation: D = V0 * T - 0.5 * A * T^2
        # Solving for initial velocity (V0) and acceleration (A)
        acceleration = -self.min_non_distractor_acceleration
        initial_velocity = (displacement - 0.5 * acceleration * time**2) / time

        return (
            initial_velocity,
            acceleration,
            initial_velocity / self.distractor_retardation_factor,
            acceleration / self.distractor_retardation_factor,
        )

    def _ideal_curve(self, similarities, num_relevant):
        min_non_distractor_displacement = num_relevant
        (
            min_non_distractor_velocity,
            min_non_distractor_acceleration,
            min_distractor_velocity,
            min_distractor_acceleration,
        ) = self._calculate_velocity_acceleration(
            min_non_distractor_displacement, num_relevant
        )

        num_distractor = len(similarities) - num_relevant

        # Create arrays for velocities and accelerations
        non_distractor_velocities = torch.full(
            (num_relevant,), min_non_distractor_velocity
        )
        non_distractor_accelerations = torch.full(
            (num_relevant,), min_non_distractor_acceleration
        )
        distractor_velocities = torch.full((num_distractor,), min_distractor_velocity)
        distractor_accelerations = torch.full(
            (num_distractor,), min_distractor_acceleration
        )

        # Integrate acceleration to get velocity
        non_distractor_velocities += torch.cumsum(non_distractor_accelerations, dim=0)
        distractor_velocities += torch.cumsum(distractor_accelerations, dim=0)

        # Combine velocities for non-distractors and distractors
        total_velocities = torch.cat((non_distractor_velocities, distractor_velocities))

        ideal_curve = total_velocities

        return ideal_curve

    def forward(self, similarities, num_relevant):
        min_non_distractor_displacement = num_relevant
        (
            min_non_distractor_velocity,
            min_non_distractor_acceleration,
            min_distractor_velocity,
            min_distractor_acceleration,
        ) = self._calculate_velocity_acceleration(
            min_non_distractor_displacement, num_relevant
        )

        # Compute first and second derivatives
        first_derivative = similarities
        second_derivative = torch.diff(
            similarities, append=torch.tensor([0.0], device=similarities.device)
        )

        # Loss for non-distractors
        non_distractor_velocity_loss = torch.sum(
            torch.relu(-first_derivative[:num_relevant] + min_non_distractor_velocity)
        )
        non_distractor_acceleration_loss = torch.sum(
            torch.relu(
                second_derivative[:num_relevant] - min_non_distractor_acceleration
            )
        )
        non_distractor_loss = (
            non_distractor_velocity_loss + non_distractor_acceleration_loss
        )

        # Loss for distractors
        distractor_velocity_loss = torch.sum(
            torch.relu(first_derivative[num_relevant:] - min_distractor_velocity)
        )
        distractor_acceleration_loss = torch.sum(
            torch.relu(second_derivative[num_relevant:] - min_distractor_acceleration)
        )
        distractor_loss = distractor_velocity_loss + distractor_acceleration_loss

        # Total loss
        total_loss = non_distractor_loss + distractor_loss
        return total_loss
