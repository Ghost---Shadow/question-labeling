import torch
from torch.cuda.amp import autocast
from training_loop_strategies.utils import (
    average_metrics,
    compute_cutoff_gain,
    compute_dissimilarities,
    compute_search_metrics,
    pick_highest_scoring_new_document,
    record_pick,
    select_next_correct,
)
from pydash import get


def train_step(config, scaler, wrapped_model, optimizer, batch, loss_fn):
    optimizer.zero_grad()

    all_metrics = []

    enable_quality = not get(
        config, "architecture.quality_diversity.disable_quality", False
    )

    enable_diversity = not get(
        config, "architecture.quality_diversity.disable_diversity", False
    )

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
            labels_mask = torch.tensor(labels_mask, device=similarities.device)

            picked_mask = torch.zeros(
                len(flat_questions), device=similarities.device, dtype=torch.bool
            )
            labels_mask_list = [labels_mask.clone()]
            picked_mask_list = [picked_mask]
            teacher_forcing = []
            cutoff_gains = []
            actual_picks = []

            can_be_picked_set = set(no_paraphrase_relevant_question_indexes)
            num_correct_answers = len(can_be_picked_set)
            total_loss = torch.zeros([], device=similarities.device)

            for _ in range(num_correct_answers):
                current_all_labels_mask = labels_mask_list[-1]
                current_picked_mask = picked_mask_list[-1]

                dissimilarities = compute_dissimilarities(
                    document_embeddings, current_picked_mask, similarities
                )

                predictions = torch.ones_like(similarities, device=similarities.device)
                if enable_quality:
                    predictions = predictions * similarities
                if enable_diversity:
                    predictions = predictions * (1 - dissimilarities)

                # Store the highest scoring document
                actual_picks.append(
                    pick_highest_scoring_new_document(predictions, actual_picks)
                )

                labels = current_all_labels_mask.float()
                loss = loss_fn(predictions, labels)
                total_loss += loss

                next_correct = select_next_correct(
                    predictions, paraphrase_lut, can_be_picked_set, current_picked_mask
                )

                if not get(config, "eval.disable_cutoff_gains", False):
                    cutoff_gains.append(
                        compute_cutoff_gain(
                            predictions,
                            labels_mask_list[0].clone(),
                            current_picked_mask,
                            paraphrase_lut,
                        )
                    )

                record_pick(
                    next_correct,
                    can_be_picked_set,
                    paraphrase_lut,
                    labels_mask_list,
                    picked_mask_list,
                    teacher_forcing,
                )

        avg_loss = total_loss / num_correct_answers
        scaler.scale(avg_loss).backward()

        cutoff_gains = torch.tensor(cutoff_gains)

        search_metrics = compute_search_metrics(
            config,
            actual_picks,
            paraphrase_lut,
            no_paraphrase_relevant_question_indexes,
        )

        all_metrics.append(
            {
                "loss": avg_loss.item(),
                **search_metrics,
                "cutoff_gain_mean": cutoff_gains.mean().item(),
                "cutoff_gain_std": cutoff_gains.std().item(),
            }
        )

    scaler.step(optimizer)
    scaler.update()

    avg_metrics = average_metrics(all_metrics)

    return avg_metrics


def eval_step(config, scaler, wrapped_model, optimizer, batch, loss_fn):
    all_metrics = []

    enable_quality = not get(
        config, "architecture.quality_diversity.disable_quality", False
    )

    enable_diversity = not get(
        config, "architecture.quality_diversity.disable_diversity", False
    )

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
            labels_mask = torch.tensor(labels_mask, device=similarities.device)

            picked_mask = torch.zeros(
                len(flat_questions), device=similarities.device, dtype=torch.bool
            )
            labels_mask_list = [labels_mask.clone()]
            picked_mask_list = [picked_mask]
            teacher_forcing = []
            cutoff_gains = []
            search_metrics = []

            can_be_picked_set = set(no_paraphrase_relevant_question_indexes)
            num_correct_answers = len(can_be_picked_set)
            total_loss = torch.zeros([], device=similarities.device)

            for _ in range(num_correct_answers):
                current_all_labels_mask = labels_mask_list[-1]
                current_picked_mask = picked_mask_list[-1]

                dissimilarities = compute_dissimilarities(
                    document_embeddings, current_picked_mask, similarities
                )

                predictions = torch.ones_like(similarities, device=similarities.device)
                if enable_quality:
                    predictions = predictions * similarities
                if enable_diversity:
                    predictions = predictions * (1 - dissimilarities)

                labels = current_all_labels_mask.float()
                loss = loss_fn(predictions, labels)
                total_loss += loss

                next_correct = select_next_correct(
                    predictions,
                    paraphrase_lut,
                    can_be_picked_set,
                    current_picked_mask,
                )

                if not get(config, "eval.disable_cutoff_gains", False):
                    cutoff_gains.append(
                        compute_cutoff_gain(
                            predictions,
                            labels_mask_list[0].clone(),
                            current_picked_mask,
                            paraphrase_lut,
                        )
                    )

                search_metrics.append(
                    compute_search_metrics(
                        config,
                        predictions,
                        paraphrase_lut,
                        can_be_picked_set,
                    )
                )

                record_pick(
                    next_correct,
                    can_be_picked_set,
                    paraphrase_lut,
                    labels_mask_list,
                    picked_mask_list,
                    teacher_forcing,
                )

            cutoff_gains = torch.tensor(cutoff_gains)

            search_metrics = average_metrics(search_metrics)

            all_metrics.append(
                {
                    "loss": (total_loss / num_correct_answers).item(),
                    **search_metrics,
                    "cutoff_gain_mean": cutoff_gains.mean().item(),
                    "cutoff_gain_std": cutoff_gains.std().item(),
                }
            )

    avg_metrics = average_metrics(all_metrics)

    return avg_metrics
