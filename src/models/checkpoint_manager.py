import hashlib
import json
import os
import random
import re
from models import MODEL_LUT
import numpy as np
import torch
from torch.cuda.amp import GradScaler
import torch.optim as optim


def generate_md5_hash(config):
    hasher = hashlib.md5()
    buf = json.dumps(config, sort_keys=True).encode()
    hasher.update(buf)
    return hasher.hexdigest()[:5]


class CheckpointManager:
    def __init__(self, config, seed):
        self.checkpoint_dir = CheckpointManager.generate_checkpoint_dir(config, seed)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Load relevant info from config
        semantic_search_model_name = config["architecture"]["semantic_search_model"][
            "name"
        ]
        learning_rate = float(config["training"]["learning_rate"])

        # Placeholders for loading
        self.last_epoch = 0
        self.config = config
        self.wrapped_model = MODEL_LUT[semantic_search_model_name](config)
        self.scaler = GradScaler()
        self.optimizer = optim.AdamW(
            self.wrapped_model.get_all_trainable_parameters(), lr=learning_rate
        )
        self.scheduler = None  # TODO

        try:
            self.load()
        except FileNotFoundError:
            print("No existing checkpoint found. Starting from scratch.")
        else:
            print("Loaded existing checkpoint.")

    @staticmethod
    def generate_checkpoint_dir(config, seed):
        config_hash = generate_md5_hash(config)
        config_name = config["name"]
        checkpoint_dir = f"./checkpoints/{config_name}_{config_hash}/seed_{seed}/"
        return checkpoint_dir

    def save(self, epoch):
        self.last_epoch = epoch
        checkpoint_path = os.path.join(self.checkpoint_dir, f"epoch_{epoch}.pth")
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.wrapped_model.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
            "random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "torch_random_state": torch.get_rng_state(),
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

    def load(self, epoch=None):
        checkpoint_files = os.listdir(self.checkpoint_dir)
        if not checkpoint_files:
            raise FileNotFoundError("No checkpoints found in the directory.")

        if epoch is None:
            checkpoint_path = self.get_latest_checkpoint(checkpoint_files)
        else:
            checkpoint_path = os.path.join(self.checkpoint_dir, f"epoch_{epoch}.pth")
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"No checkpoint found for epoch {epoch}.")

        checkpoint = torch.load(checkpoint_path)
        self.wrapped_model.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Restoring random states
        random.setstate(checkpoint["random_state"])
        np.random.set_state(checkpoint["numpy_random_state"])
        torch.set_rng_state(checkpoint["torch_random_state"])

        if "scheduler_state_dict" in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if "scaler_state_dict" in checkpoint and self.scaler is not None:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        print(f"Checkpoint loaded from {checkpoint_path}")
        self.last_epoch = checkpoint["epoch"]
        return checkpoint["epoch"]

    def get_latest_checkpoint(self, checkpoint_files):
        checkpoint_files = sorted(
            checkpoint_files,
            key=lambda x: int(re.search(r"epoch_(\d+)", x).group(1)),
        )
        latest_checkpoint = checkpoint_files[-1]
        return os.path.join(self.checkpoint_dir, latest_checkpoint)
