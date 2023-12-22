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
