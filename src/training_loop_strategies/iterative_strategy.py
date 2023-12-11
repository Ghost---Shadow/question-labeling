import torch
from torch.cuda.amp import autocast, GradScaler

import torch
from torch.cuda.amp import autocast, GradScaler


def train_step(wrapped_model, optimizer, batch, aggregation_fn, loss_fn):
    scaler = GradScaler()

    optimizer.zero_grad()

    total_loss = 0.0
    total_recall_at_k = 0.0
    total_gradient_accumulation_steps = 0

    for relevant_sentence_indexes in batch["relevant_sentence_indexes"]:
        for _ in relevant_sentence_indexes:
            total_gradient_accumulation_steps += 1

    for question, flat_questions, selection_vector, relevant_sentence_indexes in zip(
        batch["questions"],
        batch["flat_questions"],
        batch["selection_vector"],
        batch["relevant_sentence_indexes"],
    ):
        picked_so_far = torch.zeros((len(flat_questions)), dtype=torch.bool)

        labels = torch.tensor(
            selection_vector, device=wrapped_model.device, dtype=torch.float32
        )

        correct_picks = 0
        k = len(relevant_sentence_indexes)

        for next_pick_index in relevant_sentence_indexes:
            with autocast(dtype=torch.float16):
                (
                    question_embedding,
                    document_embeddings,
                ) = wrapped_model.get_query_and_document_embeddings(
                    question, flat_questions
                )

                aggregated_query_embedding = aggregation_fn(
                    question_embedding, document_embeddings, picked_so_far
                )

                # Already normalized
                similarity = (
                    aggregated_query_embedding @ document_embeddings.T
                ).squeeze()

                # Dont compute loss on items that are already picked
                relevant_similarity = similarity[~picked_so_far]
                relevant_labels = labels[~picked_so_far]

                loss = loss_fn(relevant_similarity, relevant_labels)
                loss = loss / total_gradient_accumulation_steps

            scaler.scale(loss).backward()

            total_loss += loss.item()

            # Calculate recall@k
            similarity[picked_so_far] = 0
            max_sim_index = torch.argmax(similarity).item()
            if max_sim_index in relevant_sentence_indexes:
                correct_picks += 1

            # Pick one of the correct answers
            picked_so_far[next_pick_index] = True

        recall_at_k = correct_picks / k
        total_recall_at_k += recall_at_k

    scaler.step(optimizer)
    scaler.update()

    batch_size = len(batch["questions"])
    avg_loss = total_loss / batch_size
    avg_recall_at_k = total_recall_at_k / batch_size

    return avg_loss, avg_recall_at_k


def eval_step(wrapped_model, batch, aggregation_fn, loss_fn):
    total_loss = 0.0
    total_recall_at_k = 0.0

    with torch.no_grad():
        for (
            question,
            flat_questions,
            selection_vector,
            relevant_sentence_indexes,
        ) in zip(
            batch["questions"],
            batch["flat_questions"],
            batch["selection_vector"],
            batch["relevant_sentence_indexes"],
        ):
            picked_so_far = torch.zeros((len(flat_questions)), dtype=torch.bool)
            labels = torch.tensor(
                selection_vector, device=wrapped_model.device, dtype=torch.float32
            )

            correct_picks = 0
            k = len(relevant_sentence_indexes)

            for _ in range(k):
                (
                    question_embedding,
                    document_embeddings,
                ) = wrapped_model.get_query_and_document_embeddings(
                    question, flat_questions
                )

                aggregated_query_embedding = aggregation_fn(
                    question_embedding, document_embeddings, picked_so_far
                )

                similarity = (
                    aggregated_query_embedding @ document_embeddings.T
                ).squeeze()

                # Dont compute loss on items that are already picked
                relevant_similarity = similarity[~picked_so_far]
                relevant_labels = labels[~picked_so_far]

                loss = loss_fn(relevant_similarity, relevant_labels)
                total_loss += loss.item()

                # Check if the max similarity index is in relevant_sentence_indexes
                similarity[picked_so_far] = 0
                max_sim_index = torch.argmax(similarity).item()
                if max_sim_index in relevant_sentence_indexes:
                    correct_picks += 1

                # Update picked_so_far for iterative approach
                picked_so_far[max_sim_index] = True

            recall_at_k = correct_picks / k
            total_recall_at_k += recall_at_k

    batch_size = len(batch["questions"])
    avg_loss = total_loss / batch_size
    avg_recall_at_k = total_recall_at_k / batch_size

    return avg_loss, avg_recall_at_k


def train_step_full_precision(wrapped_model, optimizer, batch, aggregation_fn, loss_fn):
    optimizer.zero_grad()

    total_loss = 0.0

    # We want to scale the loss by number of accumulation steps to
    # prevent gradient explosion
    total_gradient_accumulation_steps = 0
    for relevant_sentence_indexes in batch["relevant_sentence_indexes"]:
        for _ in relevant_sentence_indexes:
            total_gradient_accumulation_steps += 1

    for question, flat_questions, selection_vector, relevant_sentence_indexes in zip(
        batch["questions"],
        batch["flat_questions"],
        batch["selection_vector"],
        batch["relevant_sentence_indexes"],
    ):
        picked_so_far = torch.zeros((len(flat_questions)), dtype=torch.bool)

        labels = torch.tensor(
            selection_vector, device=wrapped_model.device, dtype=torch.float32
        )

        for next_pick_index in relevant_sentence_indexes:
            (
                question_embedding,
                document_embeddings,
            ) = wrapped_model.get_query_and_document_embeddings(
                question, flat_questions
            )

            aggregated_query_embedding = aggregation_fn(
                question_embedding, document_embeddings, picked_so_far
            )

            # Already normalized
            similarity = (aggregated_query_embedding @ document_embeddings.T).squeeze()

            # Dont compute loss on items that are already picked
            relevant_similarity = similarity[~picked_so_far]
            relevant_labels = labels[~picked_so_far]

            loss = loss_fn(relevant_similarity, relevant_labels)
            loss = loss / total_gradient_accumulation_steps
            loss.backward()

            total_loss += loss.item()

            # Pick one of the correct answers
            picked_so_far[next_pick_index] = True

    optimizer.step()

    return total_loss
