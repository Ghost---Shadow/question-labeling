import torch


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


def compute_recall_non_iterative(
    predictions, no_paraphrase_relevant_question_indexes, paraphrase_lut
):
    sorted_indices = torch.argsort(predictions, descending=True)

    top_indices = sorted_indices[
        : len(no_paraphrase_relevant_question_indexes)
    ].tolist()

    num_items_found = sum(
        index in top_indices or paraphrase_lut.get(index) in top_indices
        for index in no_paraphrase_relevant_question_indexes
    )

    recall_at_1 = num_items_found / len(no_paraphrase_relevant_question_indexes)

    return recall_at_1


def compute_cutoff_gain(
    similarities, global_correct_mask, current_picked_mask, paraphrase_lut
):
    # Convert to boolean mask if not already
    global_correct_mask = global_correct_mask.bool()

    # Update global_correct_mask to mark documents as incorrect if their paraphrases have been picked
    for i, picked in enumerate(current_picked_mask):
        if picked and i in paraphrase_lut:
            paraphrase_index = paraphrase_lut[i]
            global_correct_mask[paraphrase_index] = False

    # Find the least similar correct document
    least_similar_correct = similarities[global_correct_mask].min()
    most_similar_incorrect = similarities[~global_correct_mask].max()

    return (least_similar_correct - most_similar_incorrect).item()
