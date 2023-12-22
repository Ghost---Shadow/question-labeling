import torch
from torch.cuda.amp import autocast, GradScaler
from training_loop_strategies.utils import (
    average_metrics,
    compute_loss_and_similarity,
    compute_metrics,
    compute_total_gradient_accumulation_steps,
    get_batch_documents,
)


def train_step(config, wrapped_model, optimizer, batch, aggregation_model, loss_fn):
    scaler = GradScaler()

    optimizer.zero_grad()

    total_loss = 0.0
    total_gradient_accumulation_steps = compute_total_gradient_accumulation_steps(batch)
    batch_documents = get_batch_documents(batch)

    all_metrics = []

    for question, documents, selection_vector, relevant_sentence_indexes in zip(
        batch["questions"],
        batch_documents,
        batch["selection_vector"],
        batch["relevant_sentence_indexes"],
    ):
        picked_so_far = torch.zeros((len(documents)), dtype=torch.bool)

        labels = torch.tensor(
            selection_vector, device=wrapped_model.device, dtype=torch.float32
        )

        for next_pick_index in relevant_sentence_indexes:
            with autocast(dtype=torch.float16):
                similarity, loss = compute_loss_and_similarity(
                    wrapped_model,
                    aggregation_model,
                    loss_fn,
                    question,
                    documents,
                    picked_so_far,
                    labels,
                )
                loss = loss / total_gradient_accumulation_steps

            scaler.scale(loss).backward()

            total_loss += loss.item()

            # Disable the similarities things picked so far
            similarity[picked_so_far] = 0

            # Pick one of the correct answers
            picked_so_far[next_pick_index] = True

            # Calculate recall@k, precision@k, F1@k
            for k in config["eval"]["k"]:
                metrics = compute_metrics(
                    similarity, relevant_sentence_indexes, picked_so_far, k
                )
                all_metrics.append(metrics)

    scaler.step(optimizer)
    scaler.update()

    batch_size = len(batch["questions"])
    avg_loss = total_loss / batch_size

    all_metrics = average_metrics(all_metrics)

    return {
        **all_metrics,
        "loss": avg_loss,
    }


def eval_step(config, wrapped_model, batch, aggregation_model, loss_fn):
    total_loss = 0.0

    batch_documents = get_batch_documents(batch)
    all_metrics = []

    with torch.no_grad():
        for (
            question,
            documents,
            selection_vector,
            relevant_sentence_indexes,
        ) in zip(
            batch["questions"],
            batch_documents,
            batch["selection_vector"],
            batch["relevant_sentence_indexes"],
        ):
            picked_so_far = torch.zeros((len(documents)), dtype=torch.bool)
            labels = torch.tensor(
                selection_vector, device=wrapped_model.device, dtype=torch.float32
            )

            for next_pick_index in relevant_sentence_indexes:
                similarity, loss = compute_loss_and_similarity(
                    wrapped_model,
                    aggregation_model,
                    loss_fn,
                    question,
                    documents,
                    picked_so_far,
                    labels,
                )

                total_loss += loss.item()

                # Disable the similarities things picked so far
                similarity[picked_so_far] = 0

                # Update picked_so_far for iterative approach
                picked_so_far[next_pick_index] = True

                # Calculate recall@k, precision@k, F1@k
                for k in config["eval"]["k"]:
                    metrics = compute_metrics(
                        similarity, relevant_sentence_indexes, picked_so_far, k
                    )
                    all_metrics.append(metrics)

    batch_size = len(batch["questions"])
    avg_loss = total_loss / batch_size

    all_metrics = average_metrics(all_metrics)

    return {
        **all_metrics,
        "loss": avg_loss,
    }


def train_step_full_precision(
    config, wrapped_model, optimizer, batch, aggregation_model, loss_fn
):
    optimizer.zero_grad()

    total_loss = 0.0
    all_metrics = []

    batch_documents = get_batch_documents(batch)
    total_gradient_accumulation_steps = compute_total_gradient_accumulation_steps(batch)

    for question, documents, selection_vector, relevant_sentence_indexes in zip(
        batch["questions"],
        batch_documents,
        batch["selection_vector"],
        batch["relevant_sentence_indexes"],
    ):
        picked_so_far = torch.zeros((len(documents)), dtype=torch.bool)

        labels = torch.tensor(
            selection_vector, device=wrapped_model.device, dtype=torch.float32
        )

        for next_pick_index in relevant_sentence_indexes:
            similarity, loss = compute_loss_and_similarity(
                wrapped_model,
                aggregation_model,
                loss_fn,
                question,
                documents,
                picked_so_far,
                labels,
            )
            loss = loss / total_gradient_accumulation_steps
            loss.backward()

            total_loss += loss.item()

            # Pick one of the correct answers
            picked_so_far[next_pick_index] = True

            # Disable the similarities things picked so far
            similarity[picked_so_far] = 0

            # Update picked_so_far for iterative approach
            picked_so_far[next_pick_index] = True

            # Calculate recall@k, precision@k, F1@k
            for k in config["eval"]["k"]:
                metrics = compute_metrics(
                    similarity, relevant_sentence_indexes, picked_so_far, k
                )
                all_metrics.append(metrics)

    optimizer.step()

    all_metrics = average_metrics(all_metrics)

    avg_loss = total_loss / len(batch["questions"])

    return {
        **all_metrics,
        "loss": avg_loss,
    }
