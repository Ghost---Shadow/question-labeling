import hashlib
import time
import torch
from tqdm import tqdm
from training_loop_strategies.iterative_strategy import average_metrics
import wandb


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def generate_md5_hash(file_path):
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()


def validate_one_epoch(
    config,
    validation_loader,
    wrapped_search_model,
    eval_step_fn,
    aggregation_fn,
    loss_fn,
    debug=False,
):
    wrapped_search_model.model.eval()

    total_inference_time = 0
    steps = 0
    all_metrics = []

    with torch.no_grad():
        for batch in tqdm(validation_loader, desc="Validation"):
            start_time = time.time()

            # Call eval_step function
            metrics = eval_step_fn(
                config, wrapped_search_model, batch, aggregation_fn, loss_fn
            )
            all_metrics.append(metrics)

            inference_time = time.time() - start_time
            total_inference_time += inference_time

            steps += 1

            if debug and steps == 5:
                break

    average_inference_time = total_inference_time / steps
    avg_metrics = average_metrics(all_metrics)

    # Log metrics if not in debug mode
    if not debug:
        wandb.log(
            {
                "validation": {
                    **avg_metrics,
                    "inference_time": average_inference_time,
                }
            }
        )

    return avg_metrics["loss"], average_inference_time


def train_one_epoch(
    config,
    train_loader,
    wrapped_search_model,
    optimizer,
    train_step_fn,
    loss_fn,
    aggregation_fn,
    debug,
):
    wrapped_search_model.model.train()

    total_loss = 0
    num_samples_seen = 0

    pbar = tqdm(train_loader)
    for batch in pbar:
        # Call to train_step function
        metrics = train_step_fn(
            config, wrapped_search_model, optimizer, batch, aggregation_fn, loss_fn
        )

        step_loss = metrics["loss"]
        total_loss += metrics["loss"]

        num_samples_seen += sum(
            len(relevant) for relevant in batch["relevant_sentence_indexes"]
        )

        pbar.set_description(f"Loss: {step_loss:.4f}")

        if not debug:
            # Log to wandb
            wandb.log({"train": metrics})

        if debug and num_samples_seen >= 5:
            break

    average_loss = total_loss / num_samples_seen if num_samples_seen > 0 else 0

    return average_loss
