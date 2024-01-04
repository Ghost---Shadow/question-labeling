import os
import re
import torch


class CheckpointManager:
    def __init__(self, model, optimizer, scheduler, config):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.checkpoint_dir = f"./checkpoints/{config['name']}"

        # Create checkpoint directory if it doesn't exist
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def save(self, epoch):
        checkpoint_path = os.path.join(self.checkpoint_dir, f"epoch_{epoch}.pth")
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
        }

        # Include scheduler state if it's provided
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

    def load(self, epoch=None):
        if epoch is None:
            checkpoint_files = os.listdir(self.checkpoint_dir)
            if not checkpoint_files:
                raise FileNotFoundError("No checkpoints found in the directory.")
            checkpoint_path = self.get_latest_checkpoint(checkpoint_files)
        else:
            checkpoint_path = os.path.join(self.checkpoint_dir, f"epoch_{epoch}.pth")
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"No checkpoint found for epoch {epoch}.")

        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Load scheduler state if it's provided
        if "scheduler_state_dict" in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        print(f"Checkpoint loaded from {checkpoint_path}")
        return checkpoint["epoch"]

    def get_latest_checkpoint(self, checkpoint_files):
        # Sorting checkpoints numerically by epoch
        checkpoint_files = sorted(
            checkpoint_files,
            key=lambda x: int(re.search(r"epoch_(\d+)", x).group(1)),
        )
        latest_checkpoint = checkpoint_files[-1]
        return os.path.join(self.checkpoint_dir, latest_checkpoint)
