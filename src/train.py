import argparse
from dataloaders import DATA_LOADER_LUT
from losses import LOSS_LUT
from models import MODEL_LUT
from dataloaders import DATA_LOADER_LUT
from aggregation_strategies import AGGREGATION_STRATEGY_LUT
from train_utils import set_seed, train_one_epoch, validate_one_epoch
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
    aggregation_fn = AGGREGATION_STRATEGY_LUT[aggregation_strategy_name]

    # train step function
    train_step_fn, eval_step_fn = TRAINING_LOOP_STRATEGY_LUT[training_strategy_name]

    for seed in config["training"]["seeds"]:
        set_seed(seed)

        # Load data loaders after setting seed
        print("Loading data loader")
        get_loader = DATA_LOADER_LUT[dataset_name]
        train_loader, validation_loader = get_loader(batch_size=BATCH_SIZE)

        optimizer = optim.AdamW(
            wrapped_search_model.model.parameters(), lr=learning_rate
        )

        wandb.init(
            project=config["wandb"]["project"],
            name=config["wandb"]["name"] + f"_{seed}",
            config={
                **config,
                "seed": seed,
                "total_parameters": sum(
                    p.numel() for p in wrapped_search_model.model.parameters()
                ),
            },
            mode="disabled" if debug else None,
            entity=config["wandb"].get("entity", None),
        )

        if not debug:
            print("Starting warmup validation")
            val_loss, val_recall, _ = validate_one_epoch(
                validation_loader,
                wrapped_search_model,
                eval_step_fn,
                aggregation_fn,
                loss_fn,
                debug,
            )

        for epoch in range(EPOCHS):
            print(f"Start training for epoch {epoch}")
            train_loss, train_recall = train_one_epoch(
                train_loader,
                wrapped_search_model,
                optimizer,
                train_step_fn,
                loss_fn,
                aggregation_fn,
                debug,
            )
            print("Starting validation")
            val_loss, val_recall, _ = validate_one_epoch(
                validation_loader,
                wrapped_search_model,
                eval_step_fn,
                aggregation_fn,
                loss_fn,
                debug,
            )

            print(
                f"Epoch {epoch} Train Loss: {train_loss} Train recall: {train_recall} Validation Loss {val_loss} Validation recall {val_recall}"
            )

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

    main(config, debug)
