import json
from losses.triplet_loss import TripletLoss
from models.wrapped_sentence_transformer import WrappedSentenceTransformerModel
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn


def load_paraphrased_row():
    with open("src/one_off_experiments/paraphrase_sample.json") as f:
        item = json.load(f)

    flat_questions = []
    flat_questions_paraphrased = []

    index_lut = {}
    for title, questions in zip(
        item["context"]["title"],
        item["context"]["questions"],
    ):
        sent_counter = 0

        for question in questions:
            index_lut[(title, sent_counter)] = len(flat_questions)
            flat_questions.append(question)
            sent_counter += 1

    relevant_sentence_indexes = []
    for title, sent_id in zip(
        item["supporting_facts"]["title"], item["supporting_facts"]["sent_id"]
    ):
        flat_index = index_lut[(title, sent_id)]
        relevant_sentence_indexes.append(flat_index)

    selection_vector = [0] * len(flat_questions)
    for index in relevant_sentence_indexes:
        selection_vector[index] = 1

    flat_questions_paraphrased = item["context"]["questions_paraphrased"]
    paraphrased_vector = [1] * len(flat_questions_paraphrased)

    paraphrase_lut = {}
    for right, left in enumerate(relevant_sentence_indexes):
        right = right + len(flat_questions)
        paraphrase_lut[left] = right
        paraphrase_lut[right] = left

    print(paraphrase_lut.keys())

    all_questions = [*flat_questions, *flat_questions_paraphrased]
    all_selection_vector = [*selection_vector, *paraphrased_vector]

    all_selection_vector = torch.tensor(all_selection_vector, device="cuda:0")

    query = item["question"]

    left_selection_vector = all_selection_vector.clone()
    left_selection_vector[len(selection_vector) :] = 0

    right_selection_vector = all_selection_vector.clone()
    right_selection_vector[: len(selection_vector)] = 0

    assert left_selection_vector.sum() == right_selection_vector.sum()

    left_selection_vector = left_selection_vector.bool()
    right_selection_vector = right_selection_vector.bool()

    selection_vectors = (left_selection_vector, right_selection_vector)

    return (
        query,
        all_questions,
        all_selection_vector,
        relevant_sentence_indexes,
        paraphrase_lut,
        selection_vectors,
    )


config = {
    "architecture": {
        "semantic_search_model": {
            "checkpoint": "all-mpnet-base-v2",
            "device": "cuda:0",
        }
    }
}
model = WrappedSentenceTransformerModel(config)


optimizer = optim.AdamW(model.model.parameters(), lr=1e-5)
# criterion = nn.MSELoss()
criterion = TripletLoss({})


def compute_inward_metric(similarities, selection_vector):
    metric = criterion(similarities, selection_vector)

    return metric


def compute_outward_metric(document_embeddings, selection_vectors):
    left_selection_vector, right_selection_vector = selection_vectors

    # Extract the relevant vectors
    left_vector = document_embeddings[left_selection_vector]
    right_vector = document_embeddings[right_selection_vector]

    similarities = torch.matmul(left_vector, right_vector.T)

    e = torch.eye(similarities.size(0), device=similarities.device)

    metrics = []
    for similarity, labels in zip(similarities, e):
        metric = criterion(similarity, labels)
        metrics.append(metric)

    # Calculate the metric
    metric = torch.stack(metrics).mean()

    return metric


train_steps = 100
for step in range(train_steps):
    (
        query,
        all_questions,
        all_selection_vector,
        relevant_sentence_indexes,
        paraphrase_lut,
        selection_vectors,
    ) = load_paraphrased_row()

    query_embedding, document_embeddings = model.get_query_and_document_embeddings(
        query, all_questions
    )

    d_acc = torch.zeros_like(query_embedding)

    picked_mask = torch.zeros(len(all_questions), device="cuda:0", dtype=torch.bool)
    actual_selected_indices = []
    teacher_forcing = []
    all_inwards = []
    all_outwards = []

    recall_at_1 = 0

    original_all_selection_vector = all_selection_vector.clone()

    num_correct_answers = len(relevant_sentence_indexes)
    can_be_picked_set = set(relevant_sentence_indexes)
    for _ in range(num_correct_answers):
        similarities = torch.matmul(document_embeddings, query_embedding.T).squeeze()
        similarities = torch.clamp(similarities, min=0, max=1)

        inward = compute_inward_metric(similarities, original_all_selection_vector)
        all_inwards.append(inward)

        outward = compute_outward_metric(document_embeddings, selection_vectors)
        all_outwards.append(outward)

        if picked_mask.sum() > 0:
            dissimilarities = torch.matmul(document_embeddings, d_acc.T).squeeze()
            dissimilarities = torch.clamp(dissimilarities, min=0, max=1)
        else:
            dissimilarities = torch.zeros_like(similarities)

        predictions = similarities * (1 - dissimilarities)
        labels = all_selection_vector.float()
        local_inward = criterion(similarities, labels)
        loss = local_inward + inward + outward

        predictions = predictions.detach().cpu()
        predictions[picked_mask] = 0
        selected_index = torch.argmax(predictions).item()
        actual_selected_indices.append(selected_index)

        if (
            selected_index in can_be_picked_set
            or paraphrase_lut.get(selected_index) in can_be_picked_set
        ):
            recall_at_1 += 1

            next_correct = selected_index
        else:
            next_correct = list(can_be_picked_set)[0]

        # Remove item from pick
        if next_correct not in can_be_picked_set:
            can_be_picked_set.remove(paraphrase_lut[next_correct])
        else:
            can_be_picked_set.remove(next_correct)

        if picked_mask.sum() > 0:
            d_acc = document_embeddings[picked_mask].mean(dim=0)
            d_acc = torch.nn.functional.normalize(d_acc, dim=-1)

        offset = len(all_selection_vector) // 2
        all_selection_vector[next_correct] = 0
        all_selection_vector[paraphrase_lut[next_correct]] = 0
        picked_mask[next_correct] = 1
        teacher_forcing.append(next_correct)

    inward = torch.stack(all_inwards).mean()
    outward = torch.stack(all_outwards).mean()

    loss.backward()
    optimizer.step()

    recall_at_1 = recall_at_1 / len(relevant_sentence_indexes)
    print(
        f"inward: {inward}, outward: {outward} Loss: {loss.item()}, Recall@1: {recall_at_1}, actual_selected_indices: {actual_selected_indices}, teacher_forcing: {teacher_forcing}"
    )
