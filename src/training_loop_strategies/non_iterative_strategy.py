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


def eval_step(config, scaler, wrapped_model, optimizer, batch, loss_fn):
    all_metrics = []

    with torch.no_grad():
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

            recall_at_1 = compute_recall_non_iterative(
                predictions, no_paraphrase_relevant_question_indexes, paraphrase_lut
            )

            all_metrics.append(
                {
                    "loss": loss.item(),
                    "recall_at_1": recall_at_1,
                }
            )

    avg_metrics = average_metrics(all_metrics)

    return avg_metrics
