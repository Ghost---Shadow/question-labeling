import torch
from torch.cuda.amp import autocast
from training_loop_strategies.utils import (
    average_metrics,
    compute_metrics_non_iterative,
    compute_recall_non_iterative,
)


def train_step(config, scaler, wrapped_model, optimizer, batch, loss_fn):
    optimizer.zero_grad()

    all_metrics = []

    for (
        question,
        flat_questions,
        selection_vector,
        no_paraphrase_relevant_question_indexes,
        paraphrase_lut,
    ) in zip(
        batch["questions"],
        batch["flat_questions"],
        batch["selection_vector"],
        batch["relevant_sentence_indexes"],
        batch["paraphrase_lut"],
    ):
        with autocast(dtype=torch.float16):
            (
                query_embedding,
                document_embeddings,
            ) = wrapped_model.get_query_and_document_embeddings(
                question, flat_questions
            )

            similarities = torch.matmul(
                document_embeddings, query_embedding.T
            ).squeeze()
            similarities = torch.clamp(similarities, min=0, max=1)
            predictions = similarities
            selection_vector = torch.tensor(
                selection_vector, device=similarities.device
            )
            labels = selection_vector.float()
            loss = loss_fn(predictions, labels)

        scaler.scale(loss).backward()

        recall_at_1 = compute_recall_non_iterative(
            predictions, no_paraphrase_relevant_question_indexes, paraphrase_lut
        )

        all_metrics.append(
            {
                "loss": loss.item(),
                "recall_at_1": recall_at_1,
            }
        )

    scaler.step(optimizer)
    scaler.update()

    avg_metrics = average_metrics(all_metrics)

    return avg_metrics


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
