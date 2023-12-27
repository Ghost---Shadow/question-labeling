import torch


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


def compute_dissimilarities(document_embeddings, current_picked_mask, similarities):
    if current_picked_mask.sum() > 0:
        dissimilarities = torch.matmul(
            document_embeddings, document_embeddings[current_picked_mask].T
        )

        dissimilarities = torch.clamp(dissimilarities, min=0, max=1)

        # Find the maximum similarity for each document to any of the picked documents
        dissimilarities = torch.max(dissimilarities, dim=1)[0]
    else:
        # If no documents are picked, set the similarity to zero for all documents
        dissimilarities = torch.zeros(
            document_embeddings.shape[0], device=similarities.device
        )

    return dissimilarities


def select_next_correct(
    similarities, paraphrase_lut, recall_at_1, can_be_picked_set, selected_index
):
    if (
        selected_index in can_be_picked_set
        or paraphrase_lut.get(selected_index) in can_be_picked_set
    ):
        recall_at_1 += 1
        next_correct = selected_index
    else:
        cloned_similarities = similarities.clone().detach()
        mask = torch.full(similarities.shape, False)
        mask[list(can_be_picked_set)] = True
        cloned_similarities[~mask] = 0
        next_correct = torch.argmax(cloned_similarities).item()
        # Sometimes the probability collapses and argmax returns 0
        if next_correct not in can_be_picked_set:
            next_correct = list(can_be_picked_set)[0]

    return next_correct, recall_at_1


def record_pick(
    next_correct,
    can_be_picked_set,
    paraphrase_lut,
    current_all_selection_vector,
    all_selection_vector_list,
    current_picked_mask,
    picked_mask_list,
    teacher_forcing,
):
    # Remove item from pick
    if next_correct not in can_be_picked_set:
        # If model picked a paraphrase, then pick the normal version
        can_be_picked_set.remove(paraphrase_lut[next_correct])
    else:
        can_be_picked_set.remove(next_correct)

    next_all_selection_vector = current_all_selection_vector.clone()
    next_all_selection_vector[next_correct] = 0
    next_all_selection_vector[paraphrase_lut[next_correct]] = 0
    all_selection_vector_list.append(next_all_selection_vector)

    next_picked_mask = current_picked_mask.clone()
    next_picked_mask[next_correct] = True
    picked_mask_list.append(next_picked_mask)

    teacher_forcing.append(next_correct)

    return next_picked_mask, next_all_selection_vector


def compute_metrics_non_iterative(similarity, relevant_sentence_indexes, k):
    # Sort the similarity array while keeping track of the original indices
    sorted_indices = sorted(
        range(len(similarity)), key=lambda i: similarity[i], reverse=True
    )

    # Select the top k indices
    top_k_indices = sorted_indices[:k]

    # Count the number of relevant documents in the top k picks
    relevant_found = sum(idx in relevant_sentence_indexes for idx in top_k_indices)

    # Calculating metrics
    precision_at_k = relevant_found / k if k != 0 else 0
    recall_at_k = (
        relevant_found / len(relevant_sentence_indexes)
        if relevant_sentence_indexes
        else 0
    )
    f1_at_k = (
        2 * (precision_at_k * recall_at_k) / (precision_at_k + recall_at_k)
        if (precision_at_k + recall_at_k) != 0
        else 0
    )

    return {
        f"precision_at_{k}": precision_at_k,
        f"recall_at_{k}": recall_at_k,
        f"f1_at_{k}": f1_at_k,
    }


def compute_metrics(similarity, relevant_sentence_indexes, picked_so_far, k):
    # Sort the similarity array while keeping track of the original indices
    sorted_indices = sorted(
        range(len(similarity)), key=lambda i: similarity[i], reverse=True
    )

    # Initialize counters for relevant documents found and the number of picks
    relevant_found = 0
    picks = 0

    for idx in sorted_indices:
        if picks >= k:
            break
        if not picked_so_far[idx]:
            # Increment picks
            picks += 1
            # Check if the picked document is relevant
            if idx in relevant_sentence_indexes:
                relevant_found += 1

    # Calculating metrics
    precision_at_k = relevant_found / k if k != 0 else 0
    total_relevant = len(
        [idx for idx in relevant_sentence_indexes if not picked_so_far[idx]]
    )
    recall_at_k = relevant_found / total_relevant if total_relevant != 0 else 0
    f1_at_k = (
        2 * (precision_at_k * recall_at_k) / (precision_at_k + recall_at_k)
        if (precision_at_k + recall_at_k) != 0
        else 0
    )

    return {
        f"precision_at_{k}": precision_at_k,
        f"recall_at_{k}": recall_at_k,
        f"f1_at_{k}": f1_at_k,
    }


def average_metrics(metrics_array):
    # Initialize a dictionary to store the sum of values for each key
    sum_dict = {}

    # Iterate over each dictionary in the array
    for metrics in metrics_array:
        for key, value in metrics.items():
            # Add the value to the sum_dict, handling the case where the key might not exist yet
            if key in sum_dict:
                sum_dict[key] += value
            else:
                sum_dict[key] = value

    # Calculate the average for each key
    avg_dict = {key: value / len(metrics_array) for key, value in sum_dict.items()}

    return avg_dict
