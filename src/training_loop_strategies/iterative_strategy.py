import torch
from torch.cuda.amp import autocast
from training_loop_strategies.utils import (
    average_metrics,
    compute_cutoff_gain,
    compute_dissimilarities,
    compute_dissimilarities_streaming_gen,
    compute_search_metrics,
    pick_highest_scoring_new_document,
    record_pick,
    rerank_documents,
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

    enable_streaming = get(config, "training.streaming.enabled", False)
    streaming_batch_size = get(config, "training.streaming.batch_size", None)

    compute_embeddings_fn, inner_product_fn, compute_dissimilarities_fn = {
        True: (
            wrapped_model.get_query_and_document_embeddings_streaming,
            wrapped_model.inner_product_streaming,
            compute_dissimilarities_streaming_gen(streaming_batch_size),
        ),
        False: (
            wrapped_model.get_query_and_document_embeddings,
            wrapped_model.inner_product,
            compute_dissimilarities,
        ),
    }[enable_streaming]

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
            ) = compute_embeddings_fn(question, flat_questions)
            similarities = inner_product_fn(query_embedding, document_embeddings)

            similarities = torch.clamp(similarities, min=0, max=1)
            labels_mask = torch.tensor(labels_mask, device=similarities.device)

            picked_mask = torch.zeros(
                len(flat_questions), device=similarities.device, dtype=torch.bool
            )
            labels_mask_list = [labels_mask.clone()]
            picked_mask_list = [picked_mask]
            teacher_forcing = []
            actual_picks = []
            actual_pick_prediction = []

            can_be_picked_set = set(no_paraphrase_relevant_question_indexes)
            num_correct_answers = len(can_be_picked_set)
            num_hops = len(flat_questions)
            total_loss = torch.zeros([], device=similarities.device)

            for _ in range(num_hops):
                current_all_labels_mask = labels_mask_list[-1]
                current_picked_mask = picked_mask_list[-1]

                dissimilarities = compute_dissimilarities_fn(
                    document_embeddings, current_picked_mask, similarities
                )

                predictions = torch.ones_like(similarities, device=similarities.device)
                if enable_quality:
                    predictions = predictions * similarities
                if enable_diversity:
                    predictions = predictions * (1 - dissimilarities)

                # Store the highest scoring document
                picked_doc_idx = pick_highest_scoring_new_document(
                    predictions, actual_picks
                )
                actual_picks.append(picked_doc_idx)
                actual_pick_prediction.append(predictions[picked_doc_idx].item())

                if len(can_be_picked_set) > 0:
                    labels = current_all_labels_mask.float()
                    loss = loss_fn(predictions, labels)
                    total_loss += loss

                    # Teacher forcing if model picked incorrect document
                    next_correct = select_next_correct(
                        predictions,
                        paraphrase_lut,
                        can_be_picked_set,
                        current_picked_mask,
                    )

                    # Update bookkeeping
                    record_pick(
                        next_correct,
                        can_be_picked_set,
                        paraphrase_lut,
                        labels_mask_list,
                        picked_mask_list,
                        teacher_forcing,
                    )
                else:
                    # No training needs to be done here
                    # Just update bookkeeping and continue
                    # in order to get eval metrics
                    picked_doc_idx = actual_picks[-1]
                    next_picked_mask = current_picked_mask.clone()
                    next_picked_mask[picked_doc_idx] = True
                    picked_mask_list.append(next_picked_mask)

        avg_loss = total_loss / num_correct_answers
        scaler.scale(avg_loss).backward()

        # print_picks(
        #     actual_picks, no_paraphrase_relevant_question_indexes, paraphrase_lut
        # )

        search_metrics = compute_search_metrics(
            config,
            actual_picks,
            paraphrase_lut,
            no_paraphrase_relevant_question_indexes,
        )

        cutoff_gain = None
        if not get(config, "eval.disable_cutoff_gains", False):
            cutoff_gain = compute_cutoff_gain(
                actual_picks,
                actual_pick_prediction,
                paraphrase_lut,
                no_paraphrase_relevant_question_indexes,
            )

        all_metrics.append(
            {
                "loss": avg_loss.item(),
                **search_metrics,
                "cutoff_gain": cutoff_gain,
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
            _,
            no_paraphrase_relevant_question_indexes,
            paraphrase_lut,
        ) in zip(
            batch["questions"],
            batch["flat_questions"],
            batch["labels_mask"],
            batch["relevant_sentence_indexes"],
            batch["paraphrase_lut"],
        ):
            actual_picks, actual_pick_prediction = rerank_documents(
                wrapped_model,
                question,
                flat_questions,
                enable_quality,
                enable_diversity,
            )

            search_metrics = compute_search_metrics(
                config,
                actual_picks,
                paraphrase_lut,
                no_paraphrase_relevant_question_indexes,
            )

            cutoff_gain = None
            if not get(config, "eval.disable_cutoff_gains", False):
                cutoff_gain = compute_cutoff_gain(
                    actual_picks,
                    actual_pick_prediction,
                    paraphrase_lut,
                    no_paraphrase_relevant_question_indexes,
                )

            all_metrics.append(
                {
                    "loss": None,  # Not needed
                    **search_metrics,
                    "cutoff_gain": cutoff_gain,
                }
            )

    avg_metrics = average_metrics(all_metrics)

    return avg_metrics
