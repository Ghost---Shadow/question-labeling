import time
from dataloaders import DATA_LOADER_LUT
import torch
from tqdm import tqdm
from training_loop_strategies.iterative_strategy import average_metrics
import wandb


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def validate_one_epoch(
    config,
    validation_dataset_name,
    validation_loader,
    wrapped_model,
    scaler,
    optimizer,
    eval_step_fn,
    loss_fn,
    debug,
    samples_consumed,
):
    wrapped_model.model.eval()

    total_inference_time = 0
    all_metrics = []

    with torch.no_grad():
        for batch in tqdm(validation_loader, desc="Validation"):
            start_time = time.time()

            # Call eval_step function
            metrics = eval_step_fn(
                config, scaler, wrapped_model, optimizer, batch, loss_fn
            )
            all_metrics.append(metrics)

            inference_time = time.time() - start_time
            total_inference_time += inference_time

            if debug and len(all_metrics) >= 5:
                # if len(all_metrics) >= 5:
                break

    avg_metrics = average_metrics(all_metrics)

    average_inference_time = total_inference_time / len(all_metrics)
    avg_metrics["step_time"] = average_inference_time

    # Log metrics if not in debug mode
    if not debug:
        wandb.log(
            {"validation": {validation_dataset_name: avg_metrics}},
            step=samples_consumed,
        )


def train_one_epoch(
    config,
    train_dataset_name,
    train_loader,
    wrapped_model,
    scaler,
    optimizer,
    train_step_fn,
    loss_fn,
    debug,
    current_epoch,
):
    wrapped_model.model.train()

    num_steps = 0

    steps_per_epoch = len(train_loader)
    batch_size = config["training"]["batch_size"]

    pbar = tqdm(train_loader)
    for batch in pbar:
        start_time = time.time()
        metrics = train_step_fn(
            config, scaler, wrapped_model, optimizer, batch, loss_fn
        )
        metrics["step_time"] = time.time() - start_time

        step_loss = metrics["loss"]

        pbar.set_description(f"Loss: {step_loss:.4f}")

        num_steps += 1
        if debug and num_steps >= 5:
            # if num_steps >= 5:
            break

        global_step = current_epoch * steps_per_epoch + num_steps
        samples_consumed = global_step * batch_size

        if not debug:
            wandb.log({"train": {train_dataset_name: metrics}}, step=samples_consumed)

    return samples_consumed


def get_all_loaders(config):
    train_dataset_name = config["datasets"]["train"]
    validation_dataset_names = config["datasets"]["validation"]
    batch_size = config["training"]["batch_size"]

    get_train_loader, _ = DATA_LOADER_LUT[train_dataset_name]
    train_loader = get_train_loader(batch_size=batch_size)
    train_loaders = {train_dataset_name: train_loader}

    validation_loaders = {}
    for validation_dataset_name in validation_dataset_names:
        _, get_validation_loader = DATA_LOADER_LUT[validation_dataset_name]
        validation_loader = get_validation_loader(batch_size=batch_size)
        validation_loaders[validation_dataset_name] = validation_loader

    return train_loaders, validation_loaders
