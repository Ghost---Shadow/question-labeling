import json
from losses.triplet_loss import TripletLoss
from models.wrapped_sentence_transformer import WrappedSentenceTransformerModel
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from train_utils import set_seed
import wandb


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


def compute_inward_metric(criterion, similarities, selection_vector):
    metric = criterion(similarities, selection_vector)

    return metric


def compute_outward_metric(criterion, document_embeddings, selection_vectors):
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

    metric = torch.stack(metrics).mean()

    return metric


def train_session(seed, enable_inward, enable_outward, enable_local_inward):
    set_seed(seed)

    s = ""
    if enable_inward:
        s += "_i"
    if enable_outward:
        s += "_o"
    if enable_local_inward:
        s += "_li"

    wandb.init(
        project="paraphrase_experiment",
        name="paraphrase_experiment" + s + f"_{seed}",
        config={
            "inward": enable_inward,
            "outward": enable_outward,
            "local_inward": enable_local_inward,
            "seed": seed,
        },
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
    criterion = TripletLoss({})

    train_steps = 1000
    for _ in tqdm(range(train_steps), leave=False):
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
            similarities = torch.matmul(
                document_embeddings, query_embedding.T
            ).squeeze()
            similarities = torch.clamp(similarities, min=0, max=1)

            inward = compute_inward_metric(
                criterion, similarities, original_all_selection_vector
            )
            all_inwards.append(inward)

            outward = compute_outward_metric(
                criterion, document_embeddings, selection_vectors
            )
            all_outwards.append(outward)

            if picked_mask.sum() > 0:
                dissimilarities = torch.matmul(
                    document_embeddings, document_embeddings[picked_mask].T
                )

                dissimilarities = torch.clamp(dissimilarities, min=0, max=1)

                # Find the maximum similarity for each document to any of the picked documents
                dissimilarities = torch.max(dissimilarities, dim=1)[0]
            else:
                # If no documents are picked, set the similarity to zero for all documents
                dissimilarities = torch.zeros(
                    document_embeddings.shape[0], device=similarities.device
                )

            predictions = similarities * (1 - dissimilarities)
            labels = all_selection_vector.float()
            local_inward = criterion(predictions, labels)

            loss = torch.zeros([], device=local_inward.device)
            if enable_local_inward:
                loss = loss + local_inward

            if enable_inward:
                loss = loss + inward

            if enable_outward:
                loss = loss + outward

            wandb.log(
                {
                    "loss": loss,
                    "local_inward": local_inward,
                    "inward": inward,
                    "outward": outward,
                    "recall_at_1": recall_at_1 / (picked_mask.sum().item() + 1),
                }
            )

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

            all_selection_vector[next_correct] = 0
            all_selection_vector[paraphrase_lut[next_correct]] = 0
            picked_mask[next_correct] = True
            teacher_forcing.append(next_correct)

        # inward = torch.stack(all_inwards).mean()
        # outward = torch.stack(all_outwards).mean()

        loss.backward()
        optimizer.step()

        # recall_at_1 = recall_at_1 / len(relevant_sentence_indexes)

    wandb.finish()


if __name__ == "__main__":
    permutations = [
        [True, True, True],
        [False, True, True],
        [True, False, True],
        [False, False, True],
        [True, True, False],
        [False, True, False],
        [True, False, False],
    ]
    seeds = [42, 43, 44]
    for seed in tqdm(seeds):
        for enable_inward, enable_outward, enable_local_inward in tqdm(
            permutations, leave=False
        ):
            train_session(seed, enable_inward, enable_outward, enable_local_inward)
