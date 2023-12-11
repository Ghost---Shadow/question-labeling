import torch


def train_step(wrapped_model, optimizer, batch, aggregation_fn, loss_fn):
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
