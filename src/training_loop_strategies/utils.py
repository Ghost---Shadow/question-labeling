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


def compute_dissimilarities_streaming_gen(batch_size):
    def compute_dissimilarities_streaming(
        document_embeddings, current_picked_mask, similarities
    ):
        num_documents = document_embeddings.shape[0]
        target_device = similarities.device

        if batch_size >= num_documents:
            # No need to stream
            return compute_dissimilarities(
                document_embeddings, current_picked_mask, similarities
            )

        # Assuming similarities and current_picked_mask are already on GPU
        # Initialize dissimilarities tensor on GPU
        dissimilarities = torch.zeros(num_documents, device=target_device)

        if current_picked_mask.sum() > 0:
            # Only work with the embeddings that have been picked
            # Assuming picked_embeddings need to be on GPU
            # TODO: Stream picked_embeddings too
            picked_embeddings = document_embeddings[current_picked_mask.cpu()].to(
                target_device
            )

            # Process in batches
            for i in range(0, num_documents, batch_size):
                # Extract a batch of document embeddings and move to GPU
                batch_embeddings = document_embeddings[i : i + batch_size].to(
                    target_device
                )

                # Compute dissimilarities for the batch on GPU
                batch_dissimilarities = torch.matmul(
                    batch_embeddings, picked_embeddings.T
                )
                batch_dissimilarities = torch.clamp(batch_dissimilarities, min=0, max=1)

                # Find the maximum similarity for each document in the batch to any of the picked documents
                batch_max_dissimilarities = torch.max(batch_dissimilarities, dim=1)[0]

                # Store the results in the corresponding positions on GPU
                dissimilarities[i : i + batch_size] = batch_max_dissimilarities

        # The final result is on GPU
        return dissimilarities

    return compute_dissimilarities_streaming


def pick_highest_scoring_new_document(predictions, actual_picks):
    """
    Picks the highest scoring document that has not been picked yet.

    :param predictions: A tensor of scores for each document.
    :param actual_picks: A list of indices of documents that have already been picked.
    :return: The index of the highest scoring new document.
    """
    # Create a mask for already picked documents
    mask = torch.ones_like(predictions, dtype=torch.bool)
    mask[actual_picks] = False

    # Apply the mask to the predictions
    masked_predictions = predictions.masked_fill(~mask, float("-inf"))

    # Pick the highest scoring new document
    return masked_predictions.argmax().item()


def select_next_correct(
    predictions, paraphrase_lut, can_be_picked_set, current_picked_mask
):
    cloned_predictions = predictions.clone().detach()
    cloned_predictions[current_picked_mask] = 0
    selected_index = torch.argmax(cloned_predictions).item()

    if (
        selected_index in can_be_picked_set
        or paraphrase_lut.get(selected_index) in can_be_picked_set
    ):
        next_correct = selected_index
    else:
        mask = torch.full(predictions.shape, False)
        mask[list(can_be_picked_set)] = True
        cloned_predictions[~mask] = 0
        next_correct = torch.argmax(cloned_predictions).item()
        # Sometimes the probability collapses and argmax returns 0
        if next_correct not in can_be_picked_set:
            next_correct = list(can_be_picked_set)[0]

    return next_correct


def compute_search_metrics(
    config, ranking_indices, paraphrase_lut, relevant_doc_ids_without_paraphrase
):
    new_metrics = {}

    for k in config["eval"]["k"]:
        # Make a copy of the relevant documents set to track picked items
        remaining_relevant_docs = set(relevant_doc_ids_without_paraphrase)

        # Ensure k is not greater than the length of ranking_indices
        original_k = k
        k = min(k, len(ranking_indices))

        # Get the top k indices from the ranking indices
        top_k_indices = ranking_indices[:k]

        # Use a counter to track the number of relevant documents picked
        relevant_documents_count = 0

        for idx in top_k_indices:
            # Check if the document or its paraphrase is relevant and not already picked
            if (
                idx in remaining_relevant_docs
                or paraphrase_lut.get(idx) in remaining_relevant_docs
            ):
                relevant_documents_count += 1
                # Remove the picked document and its paraphrase from the remaining set
                remaining_relevant_docs.discard(idx)
                remaining_relevant_docs.discard(paraphrase_lut.get(idx))

        # Compute recall and precision at k
        total_relevant_items = len(relevant_doc_ids_without_paraphrase)
        recall_at_k = (
            relevant_documents_count / total_relevant_items
            if total_relevant_items > 0
            else 0
        )
        precision_at_k = relevant_documents_count / k

        # Compute F1 score
        if precision_at_k + recall_at_k > 0:
            f1_at_k = (
                2 * (precision_at_k * recall_at_k) / (precision_at_k + recall_at_k)
            )
        else:
            f1_at_k = 0

        # Update the metrics dictionary
        new_metrics[f"recall_at_{original_k}"] = recall_at_k
        new_metrics[f"precision_at_{original_k}"] = precision_at_k
        new_metrics[f"f1_at_{original_k}"] = f1_at_k

    return new_metrics


def compute_natural_search_metrics(
    ranking_indices, paraphrase_lut, relevant_doc_ids_without_paraphrase
):
    new_metrics = {}

    # Make a copy of the relevant documents set to track picked items
    remaining_relevant_docs = set(relevant_doc_ids_without_paraphrase)
    # Max value of true positive
    num_correct_answers = min(len(remaining_relevant_docs), len(ranking_indices))
    k = num_correct_answers

    # Get the top k indices from the ranking indices
    top_k_indices = ranking_indices[:k]

    # Use a counter to track the number of relevant documents picked
    true_positive = 0

    for idx in top_k_indices:
        # Check if the document or its paraphrase is relevant and not already picked
        if (
            idx in remaining_relevant_docs
            or paraphrase_lut.get(idx) in remaining_relevant_docs
        ):
            true_positive += 1
            # Remove the picked document and its paraphrase from the remaining set
            remaining_relevant_docs.discard(idx)
            remaining_relevant_docs.discard(paraphrase_lut.get(idx))

    false_positive = num_correct_answers - true_positive
    false_negative = num_correct_answers - true_positive

    if (true_positive + false_positive) > 0:
        precision = true_positive / (true_positive + false_positive)
    else:
        precision = 0

    if (true_positive + false_negative) > 0:
        recall = true_positive / (true_positive + false_negative)
    else:
        recall = 0

    if (precision + recall) > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0

    # Update the metrics dictionary
    new_metrics[f"recall"] = recall
    new_metrics[f"precision"] = precision
    new_metrics[f"f1"] = f1

    return new_metrics


def record_pick(
    next_correct,
    can_be_picked_set,
    paraphrase_lut,
    labels_mask_list,
    picked_mask_list,
    teacher_forcing,
):
    # should be python int, not torch int
    assert type(next_correct) == int

    # Remove item from pick
    paraphrase_index = paraphrase_lut[next_correct]
    if next_correct in can_be_picked_set:
        can_be_picked_set.remove(next_correct)
    elif paraphrase_index in can_be_picked_set:
        can_be_picked_set.remove(paraphrase_index)

    current_all_labels_mask = labels_mask_list[-1]
    next_all_labels_mask = current_all_labels_mask.clone()
    next_all_labels_mask[next_correct] = False
    next_all_labels_mask[paraphrase_index] = False
    labels_mask_list.append(next_all_labels_mask)

    current_picked_mask = picked_mask_list[-1]
    next_picked_mask = current_picked_mask.clone()
    next_picked_mask[next_correct] = True
    picked_mask_list.append(next_picked_mask)

    teacher_forcing.append(next_correct)


def average_metrics(metrics_array):
    # Initialize a dictionary to store the sum of values for each key
    sum_dict = {}

    # Iterate over each dictionary in the array
    for metrics in metrics_array:
        for key, value in metrics.items():
            if value is None:
                continue

            # Add the value to the sum_dict, handling the case where the key might not exist yet
            if key in sum_dict:
                sum_dict[key] += value
            else:
                sum_dict[key] = value

    # Calculate the average for each key
    avg_dict = {key: value / len(metrics_array) for key, value in sum_dict.items()}

    return avg_dict


def compute_cutoff_gain(
    actual_picks,
    actual_pick_prediction,
    paraphrase_lut,
    no_paraphrase_relevant_question_indexes,
):
    """
    Computes the difference between the prediction of the least similar correct document
    and the most similar incorrect document.

    :param actual_picks: List of indices of documents picked.
    :param actual_pick_prediction: Corresponding prediction scores for the actual picks.
    :param paraphrase_lut: Lookup table for paraphrases.
    :param no_paraphrase_relevant_question_indexes: Indices of relevant documents without considering paraphrases.
    :return: The calculated difference in predictions.
    """
    # Initialize variables to track the least similar correct and most similar incorrect
    least_similar_correct_score = float("inf")
    most_similar_incorrect_score = float("-inf")

    can_be_picked_set = set(no_paraphrase_relevant_question_indexes)

    # Iterate through each pick
    for pick, score in zip(actual_picks, actual_pick_prediction):
        if (
            pick in can_be_picked_set
            or paraphrase_lut.get(pick, pick) in can_be_picked_set
        ):
            least_similar_correct_score = min(least_similar_correct_score, score)
            can_be_picked_set.discard(pick)
            can_be_picked_set.discard(paraphrase_lut.get(pick, pick))
        else:
            most_similar_incorrect_score = max(most_similar_incorrect_score, score)

    # Calculate the difference
    return least_similar_correct_score - most_similar_incorrect_score


def compute_cutoff_gain_histogram(
    actual_picks,
    actual_pick_prediction,
    paraphrase_lut,
    no_paraphrase_relevant_question_indexes,
    resolution,
):
    predictions_so_far = [1]
    picks_so_far = []

    result = {}

    for pick, prediction in zip(actual_picks, actual_pick_prediction):
        last_prediction = predictions_so_far[-1]
        prediction = int(prediction / resolution) * resolution
        gain = last_prediction - prediction
        rounded_gain = round(gain / resolution)

        predictions_so_far.append(prediction)
        picks_so_far.append(pick)

        result[rounded_gain] = compute_natural_search_metrics(
            picks_so_far,
            paraphrase_lut,
            no_paraphrase_relevant_question_indexes,
        )
        result[rounded_gain]["real_k"] = len(picks_so_far)

    return result


def accumulate_gain_cutoff(gain_histogram_accumulator, cutoff_gain_histogram):
    for gain, metrics in cutoff_gain_histogram.items():
        assert type(gain) == int, gain  # quantized
        assert type(metrics) == dict, metrics
        if gain in gain_histogram_accumulator:
            gain_histogram_accumulator[gain].append(metrics)
        else:
            gain_histogram_accumulator[gain] = [metrics]


def rerank_documents(
    wrapped_model,
    question,
    flat_questions,
    enable_quality=True,
    enable_diversity=True,
):
    (
        query_embedding,
        document_embeddings,
    ) = wrapped_model.get_query_and_document_embeddings(question, flat_questions)

    similarities = (query_embedding @ document_embeddings.T).squeeze()
    similarities = torch.clamp(similarities, min=0, max=1)

    picked_mask = torch.zeros(
        len(flat_questions), device=similarities.device, dtype=torch.bool
    )
    picked_mask_list = [picked_mask]
    actual_picks = []
    actual_pick_prediction = []

    num_hops = len(flat_questions)

    for _ in range(num_hops):
        current_picked_mask = picked_mask_list[-1]

        predictions = torch.ones_like(similarities, device=similarities.device)
        if enable_quality:
            predictions = predictions * similarities
        if enable_diversity:
            dissimilarities = compute_dissimilarities(
                document_embeddings, current_picked_mask, similarities
            )
            predictions = predictions * (1 - dissimilarities)

        # Store the highest scoring document
        picked_doc_idx = pick_highest_scoring_new_document(predictions, actual_picks)
        actual_picks.append(picked_doc_idx)
        actual_pick_prediction.append(predictions[picked_doc_idx].item())

        picked_doc_idx = actual_picks[-1]
        next_picked_mask = current_picked_mask.clone()
        next_picked_mask[picked_doc_idx] = True
        picked_mask_list.append(next_picked_mask)

    return actual_picks, actual_pick_prediction


def print_picks(actual_picks, no_paraphrase_relevant_question_indexes, paraphrase_lut):
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RESET = "\033[0m"
    already_picked = set()

    for pick in actual_picks:
        # Determine the color based on the conditions
        if pick in already_picked:
            color = YELLOW
        elif (
            pick in no_paraphrase_relevant_question_indexes
            or paraphrase_lut.get(pick) in no_paraphrase_relevant_question_indexes
        ):
            color = GREEN
        else:
            color = ""

        print(f"{color}{pick}{RESET}", end=" ")

        # Add pick and its paraphrase (if exists) to already_picked
        already_picked.add(pick)
        if pick in paraphrase_lut:
            already_picked.add(paraphrase_lut[pick])

    print()  # For a new line after printing all picks
