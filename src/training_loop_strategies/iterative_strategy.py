import torch
from torch.cuda.amp import autocast
from training_loop_strategies.utils import (
    average_metrics,
    compute_dissimilarities,
    compute_loss_and_similarity,
    compute_metrics,
    record_pick,
    select_next_correct,
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
            selection_vector = torch.tensor(
                selection_vector, device=similarities.device
            )

            picked_mask = torch.zeros(
                len(flat_questions), device="cuda:0", dtype=torch.bool
            )
            selection_vector_list = [selection_vector.clone()]
            picked_mask_list = [picked_mask]
            teacher_forcing = []
            recall_at_1 = 0

            can_be_picked_set = set(no_paraphrase_relevant_question_indexes)
            num_correct_answers = len(can_be_picked_set)
            total_loss = torch.zeros([], device=similarities.device)

            for _ in range(num_correct_answers):
                current_all_selection_vector = selection_vector_list[-1]
                current_picked_mask = picked_mask_list[-1]

                dissimilarities = compute_dissimilarities(
                    document_embeddings, current_picked_mask, similarities
                )

                predictions = similarities * (1 - dissimilarities)
                labels = current_all_selection_vector.float()
                loss = loss_fn(predictions, labels)
                total_loss += loss

                cloned_predictions = predictions.clone()
                cloned_predictions[current_picked_mask] = 0
                selected_index = torch.argmax(cloned_predictions).item()

                next_correct, recall_at_1 = select_next_correct(
                    similarities,
                    paraphrase_lut,
                    recall_at_1,
                    can_be_picked_set,
                    selected_index,
                )

                record_pick(
                    next_correct,
                    can_be_picked_set,
                    paraphrase_lut,
                    current_all_selection_vector,
                    selection_vector_list,
                    current_picked_mask,
                    picked_mask_list,
                    teacher_forcing,
                )

        scaler.scale(total_loss).backward()

        all_metrics.append(
            {
                "loss": (total_loss / num_correct_answers).item(),
                "recall_at_1": recall_at_1 / num_correct_answers,
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


def train_step_full_precision(config, scaler, wrapped_model, optimizer, batch, loss_fn):
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
        (
            query_embedding,
            document_embeddings,
        ) = wrapped_model.get_query_and_document_embeddings(question, flat_questions)

        similarities = torch.matmul(document_embeddings, query_embedding.T).squeeze()
        similarities = torch.clamp(similarities, min=0, max=1)
        selection_vector = torch.tensor(selection_vector, device=similarities.device)

        picked_mask = torch.zeros(
            len(flat_questions), device="cuda:0", dtype=torch.bool
        )
        selection_vector_list = [selection_vector.clone()]
        picked_mask_list = [picked_mask]
        teacher_forcing = []
        recall_at_1 = 0

        can_be_picked_set = set(no_paraphrase_relevant_question_indexes)
        num_correct_answers = len(can_be_picked_set)
        total_loss = torch.zeros([], device=similarities.device)

        for _ in range(num_correct_answers):
            current_all_selection_vector = selection_vector_list[-1]
            current_picked_mask = picked_mask_list[-1]

            dissimilarities = compute_dissimilarities(
                document_embeddings, current_picked_mask, similarities
            )

            predictions = similarities * (1 - dissimilarities)
            labels = current_all_selection_vector.float()
            loss = loss_fn(predictions, labels)
            total_loss += loss

            cloned_predictions = predictions.clone()
            cloned_predictions[current_picked_mask] = 0
            selected_index = torch.argmax(cloned_predictions).item()

            next_correct, recall_at_1 = select_next_correct(
                similarities,
                paraphrase_lut,
                recall_at_1,
                can_be_picked_set,
                selected_index,
            )

            record_pick(
                next_correct,
                can_be_picked_set,
                paraphrase_lut,
                current_all_selection_vector,
                selection_vector_list,
                current_picked_mask,
                picked_mask_list,
                teacher_forcing,
            )

        total_loss.backward()

        all_metrics.append(
            {
                "loss": (total_loss / num_correct_answers).item(),
                "recall_at_1": recall_at_1 / num_correct_answers,
            }
        )

    optimizer.step()

    avg_metrics = average_metrics(all_metrics)

    return avg_metrics
