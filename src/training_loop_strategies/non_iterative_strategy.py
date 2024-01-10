import torch
from torch.cuda.amp import autocast
from training_loop_strategies.utils import (
    average_metrics,
    compute_search_metrics,
)


def train_step(config, scaler, wrapped_model, optimizer, batch, loss_fn):
    optimizer.zero_grad()

    all_metrics = []

    for (
        question,
        flat_questions,
        labels_mask,
        no_paraphrase_relevant_question_indexes,
        paraphrase_lut,
    ) in zip(
        batch["questions"],
        batch["flat_questions"],
        batch["labels_mask"],
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
            labels_mask = torch.tensor(labels_mask, device=similarities.device)
            labels = labels_mask.float()
            loss = loss_fn(predictions, labels)

        scaler.scale(loss).backward()

        ranking_indices = predictions.argsort(descending=True).tolist()

        search_metrics = compute_search_metrics(
            config,
            ranking_indices,
            paraphrase_lut,
            no_paraphrase_relevant_question_indexes,
        )

        all_metrics.append(
            {
                "loss": loss.item(),
                **search_metrics,
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
            labels_mask,
            no_paraphrase_relevant_question_indexes,
            paraphrase_lut,
        ) in zip(
            batch["questions"],
            batch["flat_questions"],
            batch["labels_mask"],
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
            labels_mask = torch.tensor(labels_mask, device=similarities.device)
            labels = labels_mask.float()
            loss = loss_fn(predictions, labels)

            ranking_indices = predictions.argsort(descending=True).tolist()

            search_metrics = compute_search_metrics(
                config,
                ranking_indices,
                paraphrase_lut,
                no_paraphrase_relevant_question_indexes,
            )

            all_metrics.append(
                {
                    "loss": loss.item(),
                    **search_metrics,
                }
            )

    avg_metrics = average_metrics(all_metrics)

    return avg_metrics
