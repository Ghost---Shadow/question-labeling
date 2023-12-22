import torch
from torch.cuda.amp import autocast, GradScaler


def compute_loss_and_similarity(
    wrapped_model,
    aggregation_model,
    loss_fn,
    question,
    documents,
    picked_so_far,
    labels,
):
    (
        question_embedding,
        document_embeddings,
    ) = wrapped_model.get_query_and_document_embeddings(question, documents)

    aggregated_query_embedding = aggregation_model(
        question_embedding, document_embeddings, picked_so_far
    )

    # Already normalized
    similarity = (aggregated_query_embedding @ document_embeddings.T).squeeze()

    # Dont compute loss on items that are already picked
    relevant_similarity = similarity[~picked_so_far]
    relevant_labels = labels[~picked_so_far]

    loss = loss_fn(relevant_similarity, relevant_labels)
    return similarity, loss


def get_batch_documents(batch):
    # If synthetic questions exist, use them.
    # Otherwise fallback to original dataset
    if "flat_questions" in batch:
        batch_documents = batch["flat_questions"]
    else:
        batch_documents = batch["flat_sentences"]

    return batch_documents


def compute_total_gradient_accumulation_steps(batch):
    # We want to scale the loss by number of accumulation steps to
    # prevent gradient explosion
    total_gradient_accumulation_steps = 0
    for relevant_sentence_indexes in batch["relevant_sentence_indexes"]:
        for _ in relevant_sentence_indexes:
            total_gradient_accumulation_steps += 1

    return total_gradient_accumulation_steps


def train_step(wrapped_model, optimizer, batch, aggregation_model, loss_fn):
    scaler = GradScaler()

    optimizer.zero_grad()

    total_loss = 0.0
    total_gradient_accumulation_steps = compute_total_gradient_accumulation_steps(batch)
    batch_documents = get_batch_documents(batch)

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

            # TODO: Calculate recall@k, precision@k, F1@k

    scaler.step(optimizer)
    scaler.update()

    batch_size = len(batch["questions"])
    avg_loss = total_loss / batch_size

    return avg_loss, 0


def eval_step(wrapped_model, batch, aggregation_model, loss_fn):
    total_loss = 0.0

    batch_documents = get_batch_documents(batch)

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

                # TODO: Calculate recall@k, precision@k, F1@k

    batch_size = len(batch["questions"])
    avg_loss = total_loss / batch_size

    return avg_loss, 0


def train_step_full_precision(
    wrapped_model, optimizer, batch, aggregation_model, loss_fn
):
    optimizer.zero_grad()

    total_loss = 0.0

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

            # TODO: Calculate recall@k, precision@k, F1@k

    optimizer.step()

    return total_loss
