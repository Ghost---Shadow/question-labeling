import torch
from torch.cuda.amp import autocast, GradScaler
from training_loop_strategies.utils import (
    average_metrics,
    compute_metrics_non_iterative,
    get_batch_documents,
)


def train_step(config, wrapped_model, optimizer, batch, aggregation_model, loss_fn):
    scaler = GradScaler()

    optimizer.zero_grad()

    total_loss = 0.0
    batch_documents = get_batch_documents(batch)

    all_metrics = []

    for question, documents, selection_vector, relevant_sentence_indexes in zip(
        batch["questions"],
        batch_documents,
        batch["selection_vector"],
        batch["relevant_sentence_indexes"],
    ):
        labels = torch.tensor(
            selection_vector, device=wrapped_model.device, dtype=torch.float32
        )

        with autocast(dtype=torch.float16):
            (
                question_embedding,
                document_embeddings,
            ) = wrapped_model.get_query_and_document_embeddings(question, documents)

            similarity = (question_embedding @ document_embeddings.T).squeeze()

            loss = loss_fn(similarity, labels)

        scaler.scale(loss).backward()

        total_loss += loss.item()

        # Calculate recall@k, precision@k, F1@k
        for k in config["eval"]["k"]:
            metrics = compute_metrics_non_iterative(
                similarity, relevant_sentence_indexes, k
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
            labels = torch.tensor(
                selection_vector, device=wrapped_model.device, dtype=torch.float32
            )

            (
                question_embedding,
                document_embeddings,
            ) = wrapped_model.get_query_and_document_embeddings(question, documents)

            similarity = (question_embedding @ document_embeddings.T).squeeze()

            loss = loss_fn(similarity, labels)

            total_loss += loss.item()

            # Calculate recall@k, precision@k, F1@k
            for k in config["eval"]["k"]:
                metrics = compute_metrics_non_iterative(
                    similarity, relevant_sentence_indexes, k
                )
                all_metrics.append(metrics)

    batch_size = len(batch["questions"])
    avg_loss = total_loss / batch_size

    all_metrics = average_metrics(all_metrics)

    return {
        **all_metrics,
        "loss": avg_loss,
    }
