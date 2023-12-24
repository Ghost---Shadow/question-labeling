import argparse
import os
from pathlib import Path
from dataloaders import DATA_LOADER_LUT
from losses import LOSS_LUT
from models import MODEL_LUT
from dataloaders import DATA_LOADER_LUT
from aggregation_strategies import AGGREGATION_STRATEGY_LUT
from train_utils import generate_md5_hash, set_seed, train_one_epoch, validate_one_epoch
from training_loop_strategies import TRAINING_LOOP_STRATEGY_LUT
import yaml
import torch.optim as optim
import wandb


def main(config, debug):
    # TODO: recall is batch size dependent
    BATCH_SIZE = config["training"]["batch_size"]
    EPOCHS = config["training"]["epochs"]
    learning_rate = float(config["training"]["learning_rate"])

    dataset_name = config["dataset"]["name"]
    semantic_search_model_name = config["architecture"]["semantic_search_model"]["name"]
    loss_name = config["architecture"]["loss"]["name"]
    aggregation_strategy_name = config["architecture"]["aggregation_strategy"]["name"]
    training_strategy_name = config["training"]["strategy"]["name"]

    # Models
    print("Loading model")
    wrapped_search_model = MODEL_LUT[semantic_search_model_name](config)

    # loss function
    loss_fn = LOSS_LUT[loss_name](config)

    # aggegration function
    aggregation_model = AGGREGATION_STRATEGY_LUT[aggregation_strategy_name](
        config, wrapped_search_model
    )

    # train step function
    train_step_fn, eval_step_fn = TRAINING_LOOP_STRATEGY_LUT[training_strategy_name]

    for seed in config["training"]["seeds"]:
        set_seed(seed)

        # Load data loaders after setting seed
        print("Loading data loader")
        get_loader = DATA_LOADER_LUT[dataset_name]
        train_loader, validation_loader = get_loader(batch_size=BATCH_SIZE)

        optimizer = optim.AdamW(
            wrapped_search_model.get_all_trainable_parameters(), lr=learning_rate
        )

        wandb.init(
            project=config["wandb"]["project"],
            name=config["wandb"]["name"] + f"_{seed}",
            config={
                **config,
                "seed": seed,
                "total_parameters": sum(
                    p.numel()
                    for p in wrapped_search_model.get_all_trainable_parameters()
                ),
            },
            mode="disabled" if debug else None,
            entity=config["wandb"].get("entity", None),
        )

        if not debug:
            print("Starting warmup validation")
            val_loss, _ = validate_one_epoch(
                config,
                validation_loader,
                wrapped_search_model,
                eval_step_fn,
                aggregation_model,
                loss_fn,
                debug,
            )

        for epoch in range(EPOCHS):
            print(f"Start training for epoch {epoch}")
            train_loss = train_one_epoch(
                config,
                train_loader,
                wrapped_search_model,
                optimizer,
                train_step_fn,
                loss_fn,
                aggregation_model,
                debug,
            )
            print("Starting validation")
            val_loss, _ = validate_one_epoch(
                config,
                validation_loader,
                wrapped_search_model,
                eval_step_fn,
                aggregation_model,
                loss_fn,
                debug,
            )

            print(f"Epoch {epoch} Train Loss: {train_loss} Validation Loss {val_loss}")

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
    if os.path.exists(file_path):
        print(f"File {file_path} already exists. Skipping main execution.")
    else:
        main(config, debug)
        # After main execution, create the file to indicate completion
        with open(file_path, "w") as file:
            file.write("Completed")
