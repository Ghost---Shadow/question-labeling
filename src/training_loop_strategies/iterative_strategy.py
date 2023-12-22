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


def train_step(config, wrapped_model, optimizer, batch, aggregation_model, loss_fn):
    scaler = GradScaler()

    optimizer.zero_grad()

    total_loss = 0.0
    total_gradient_accumulation_steps = compute_total_gradient_accumulation_steps(batch)
    batch_documents = get_batch_documents(batch)

    all_metrics = []

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

            # Calculate recall@k, precision@k, F1@k
            for k in config["eval"]["k"]:
                metrics = compute_metrics(
                    similarity, relevant_sentence_indexes, picked_so_far, k
                )
                all_metrics.append(metrics)

    scaler.step(optimizer)
    scaler.update()

    batch_size = len(batch["questions"])
    avg_loss = total_loss / batch_size

    all_metrics = average_metrics(all_metrics)

    return {
        **all_metrics,
        "loss": avg_loss,
    }


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


def train_step_full_precision(
    config, wrapped_model, optimizer, batch, aggregation_model, loss_fn
):
    optimizer.zero_grad()

    total_loss = 0.0
    all_metrics = []

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

            # Calculate recall@k, precision@k, F1@k
            for k in config["eval"]["k"]:
                metrics = compute_metrics(
                    similarity, relevant_sentence_indexes, picked_so_far, k
                )
                all_metrics.append(metrics)

    optimizer.step()

    all_metrics = average_metrics(all_metrics)

    avg_loss = total_loss / len(batch["questions"])

    return {
        **all_metrics,
        "loss": avg_loss,
    }
