import argparse
from pathlib import Path
from losses import LOSS_LUT
from models.checkpoint_manager import CheckpointManager
from train_utils import (
    get_all_loaders,
    set_seed,
    train_one_epoch,
    validate_one_epoch,
)
from training_loop_strategies import TRAINING_LOOP_STRATEGY_LUT
import yaml
import wandb


def main(config, debug):
    EPOCHS = config["training"]["epochs"]

    loss_name = config["architecture"]["loss"]["name"]
    training_strategy_name = config["training"]["strategy"]["name"]

    # loss function
    loss_fn = LOSS_LUT[loss_name](config)

    # train step function
    train_step_fn, eval_step_fn = TRAINING_LOOP_STRATEGY_LUT[training_strategy_name]

    for seed in config["training"]["seeds"]:
        set_seed(seed)

        # Load data loaders after setting seed
        print("Loading data loader")
        train_loaders, validation_loaders = get_all_loaders(config)

        # Checkpoint manager
        checkpoint_manager = CheckpointManager(config, seed)

        # This seed is sweeped already
        if checkpoint_manager.last_epoch >= EPOCHS:
            continue

        if not debug:
            wandb.init(
                project=config["wandb"]["project"],
                name=config["wandb"]["name"] + f"_{seed}",
                config={
                    **config,
                    "seed": seed,
                    "total_parameters": sum(
                        p.numel()
                        for p in checkpoint_manager.wrapped_model.get_all_trainable_parameters()
                    ),
                },
                mode="disabled" if debug else None,
                entity=config["wandb"].get("entity", None),
                resume=True,
            )

        if not debug and checkpoint_manager.last_epoch == -1:
            print("Starting warmup validation")
            for validation_dataset_name in validation_loaders:
                validation_loader = validation_loaders[validation_dataset_name]
                validate_one_epoch(
                    config,
                    validation_dataset_name,
                    validation_loader,
                    checkpoint_manager.wrapped_model,
                    checkpoint_manager.scaler,
                    checkpoint_manager.optimizer,
                    eval_step_fn,
                    loss_fn,
                    debug,
                )

        for epoch in range(checkpoint_manager.last_epoch + 1, EPOCHS):
            print(f"Start training for epoch {epoch}")
            for train_dataset_name in train_loaders:
                train_loader = train_loaders[train_dataset_name]
                train_loss = train_one_epoch(
                    config,
                    train_dataset_name,
                    train_loader,
                    checkpoint_manager.wrapped_model,
                    checkpoint_manager.scaler,
                    checkpoint_manager.optimizer,
                    train_step_fn,
                    loss_fn,
                    debug,
                )
            print("Starting validation")
            total_val_loss = 0
            for validation_dataset_name in validation_loaders:
                validation_loader = validation_loaders[validation_dataset_name]
                val_loss, _ = validate_one_epoch(
                    config,
                    validation_dataset_name,
                    validation_loader,
                    checkpoint_manager.wrapped_model,
                    checkpoint_manager.scaler,
                    checkpoint_manager.optimizer,
                    eval_step_fn,
                    loss_fn,
                    debug,
                )
                total_val_loss += val_loss
            avg_val_loss = total_val_loss / len(validation_loaders)

            print(
                f"Epoch {epoch} Train Loss: {train_loss} Validation Loss {avg_val_loss}"
            )

            if not debug:
                checkpoint_manager.save(epoch)

        if not debug:
            wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a model based on provided config"
    )
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    debug = args.debug

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    assert (
        Path(args.config).stem == config["wandb"]["name"]
    ), "Filename and config.wandb.name does not match"

    main(config, debug)
