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

    selected_indices = []

    recall_at_1 = 0

    for next_correct in relevant_sentence_indexes:
        similarities = torch.matmul(document_embeddings, query_embedding.T).squeeze()
        similarities = torch.clamp(similarities, min=0, max=1)
        similarities[selected_indices] = 0

        if len(selected_indices) > 0:
            dissimilarities = torch.matmul(document_embeddings, d_acc.T).squeeze()
        else:
            dissimilarities = torch.zeros_like(similarities)
        dissimilarities[selected_indices] = 1

        predictions = similarities * (1 - dissimilarities)
        labels = all_selection_vector.float()
        loss = criterion(predictions, labels)

        selected_index = torch.argmax(predictions).item()
        if (
            selected_index in relevant_sentence_indexes
            or paraphrase_lut.get(selected_index) in relevant_sentence_indexes
        ):
            recall_at_1 += 1

        if len(selected_indices) > 0:
            d_acc = (d_acc + document_embeddings[next_correct]) / 2
            d_acc = torch.nn.functional.normalize(d_acc, dim=-1)

        offset = len(all_selection_vector) // 2
        all_selection_vector[next_correct] = 0
        all_selection_vector[paraphrase_lut[next_correct]] = 0
        selected_indices.append(next_correct)

    loss.backward()
    optimizer.step()

    recall_at_1 = recall_at_1 / len(relevant_sentence_indexes)
    print(f"Step {step+1}, Loss: {loss.item()}, Recall@1: {recall_at_1}")
