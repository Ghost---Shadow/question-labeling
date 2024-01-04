import argparse
import os
from pathlib import Path
from losses import LOSS_LUT
from models import MODEL_LUT
from train_utils import (
    generate_md5_hash,
    get_all_loaders,
    set_seed,
    train_one_epoch,
    validate_one_epoch,
)
from training_loop_strategies import TRAINING_LOOP_STRATEGY_LUT
import yaml
import torch.optim as optim
import wandb
from torch.cuda.amp import GradScaler


def main(config, debug):
    EPOCHS = config["training"]["epochs"]
    learning_rate = float(config["training"]["learning_rate"])

    semantic_search_model_name = config["architecture"]["semantic_search_model"]["name"]
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

        # Models
        print("Loading model")
        wrapped_model = MODEL_LUT[semantic_search_model_name](config)

        optimizer = optim.AdamW(
            wrapped_model.get_all_trainable_parameters(), lr=learning_rate
        )
        scaler = GradScaler()

        if not debug:
            wandb.init(
                project=config["wandb"]["project"],
                name=config["wandb"]["name"] + f"_{seed}",
                config={
                    **config,
                    "seed": seed,
                    "total_parameters": sum(
                        p.numel() for p in wrapped_model.get_all_trainable_parameters()
                    ),
                },
                mode="disabled" if debug else None,
                entity=config["wandb"].get("entity", None),
            )

        if not debug:
            print("Starting warmup validation")
            for validation_dataset_name in validation_loaders:
                validation_loader = validation_loaders[validation_dataset_name]
                validate_one_epoch(
                    config,
                    validation_dataset_name,
                    validation_loader,
                    wrapped_model,
                    scaler,
                    optimizer,
                    eval_step_fn,
                    loss_fn,
                    debug,
                )

        for epoch in range(EPOCHS):
            print(f"Start training for epoch {epoch}")
            for train_dataset_name in train_loaders:
                train_loader = train_loaders[train_dataset_name]
                train_loss = train_one_epoch(
                    config,
                    train_dataset_name,
                    train_loader,
                    wrapped_model,
                    scaler,
                    optimizer,
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
                    wrapped_model,
                    scaler,
                    optimizer,
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

    # Generate MD5 hash of the config file
    config_file_name = Path(args.config).stem
    hash_value = generate_md5_hash(args.config)
    file_path = f"./experiments/completions/{config_file_name}_{hash_value}.done"

    # Check if the file exists
    if os.path.exists(file_path) and not debug:
        print(f"File {file_path} already exists. Skipping main execution.")
    else:
        main(config, debug)

        if not debug:
            # After main execution, create the file to indicate completion
            with open(file_path, "w") as file:
                file.write("Completed")
