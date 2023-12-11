import time
import torch
from tqdm import tqdm
from torch.cuda.amp import autocast
import wandb


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def validate_one_epoch(
    validation_loader,
    wrapped_search_model,
    eval_step_fn,
    aggregation_fn,
    loss_fn,
    debug=False,
):
    wrapped_search_model.model.eval()

    total_val_loss = 0
    total_recall_at_k = 0
    num_samples_seen = 0
    total_inference_time = 0
    steps = 0

    with torch.no_grad():
        for batch in tqdm(validation_loader, desc="Validation"):
            start_time = time.time()

            # Call eval_step function
            loss, recall_at_k = eval_step_fn(
                wrapped_search_model, batch, aggregation_fn, loss_fn
            )

            inference_time = time.time() - start_time
            total_inference_time += inference_time

            total_val_loss += loss
            total_recall_at_k += recall_at_k
            num_samples_seen += len(batch["questions"])
            steps += 1

            if debug and steps == 5:
                break

    val_loss = total_val_loss / steps
    val_recall_at_k = total_recall_at_k / steps
    average_inference_time = total_inference_time / steps

    # Log metrics if not in debug mode
    if not debug:
        wandb.log(
            {
                "validation": {
                    "loss": val_loss,
                    "recall_at_k": val_recall_at_k,
                    "inference_time": average_inference_time,
                }
            }
        )

    return val_loss, val_recall_at_k, average_inference_time


def train_one_epoch(
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
    total_recall = 0
    num_samples_seen = 0

    pbar = tqdm(train_loader)
    for batch in pbar:
        # Call to train_step function
        step_loss, step_recall = train_step_fn(
            wrapped_search_model, optimizer, batch, aggregation_fn, loss_fn
        )

        total_loss += step_loss
        total_recall += step_recall

        num_samples_seen += sum(
            len(relevant) for relevant in batch["relevant_sentence_indexes"]
        )

        pbar.set_description(f"Loss: {step_loss:.4f}")

        if not debug:
            # Log to wandb
            wandb.log(
                {
                    "train": {
                        "loss": step_loss,
                        "recall": step_recall,
                    },
                }
            )

        if debug and num_samples_seen >= 5:
            break

    average_loss = total_loss / num_samples_seen if num_samples_seen > 0 else 0
    average_recall = total_recall / num_samples_seen if num_samples_seen > 0 else 0

    return average_loss, average_recall
