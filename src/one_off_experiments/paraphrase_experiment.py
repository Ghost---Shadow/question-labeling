import json
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

    all_questions = [*flat_questions, *flat_questions_paraphrased]
    all_selection_vector = [*selection_vector, *paraphrased_vector]

    all_selection_vector = torch.tensor(all_selection_vector, device="cuda:0")

    query = item["question"]

    return (
        query,
        all_questions,
        all_selection_vector,
        relevant_sentence_indexes,
        paraphrase_lut,
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
criterion = nn.MSELoss()

train_steps = 100
for step in range(train_steps):
    (
        query,
        all_questions,
        all_selection_vector,
        relevant_sentence_indexes,
        paraphrase_lut,
    ) = load_paraphrased_row()

    query_embedding, document_embeddings = model.get_query_and_document_embeddings(
        query, all_questions
    )

    d_acc = torch.zeros_like(query_embedding)

    picked_mask = torch.zeros(len(all_questions), device="cuda:0", dtype=torch.bool)
    actual_selected_indices = []
    teacher_forcing = []

    recall_at_1 = 0

    num_correct_answers = len(relevant_sentence_indexes)
    can_be_picked_set = set(relevant_sentence_indexes)
    for _ in range(num_correct_answers):
        similarities = torch.matmul(document_embeddings, query_embedding.T).squeeze()
        similarities = torch.clamp(similarities, min=0, max=1)

        if picked_mask.sum() > 0:
            dissimilarities = torch.matmul(document_embeddings, d_acc.T).squeeze()
            dissimilarities = torch.clamp(dissimilarities, min=0, max=1)
        else:
            dissimilarities = torch.zeros_like(similarities)

        predictions = similarities * (1 - dissimilarities)
        labels = all_selection_vector.float()
        loss = criterion(predictions, labels)

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
            d_acc = (d_acc + document_embeddings[next_correct]) / 2
            d_acc = torch.nn.functional.normalize(d_acc, dim=-1)

        offset = len(all_selection_vector) // 2
        all_selection_vector[next_correct] = 0
        all_selection_vector[paraphrase_lut[next_correct]] = 0
        picked_mask[next_correct] = 1
        teacher_forcing.append(next_correct)

    loss.backward()
    optimizer.step()

    recall_at_1 = recall_at_1 / len(relevant_sentence_indexes)
    print(
        f"Step {step+1}, Loss: {loss.item()}, Recall@1: {recall_at_1}, actual_selected_indices: {actual_selected_indices}, teacher_forcing: {teacher_forcing}"
    )
