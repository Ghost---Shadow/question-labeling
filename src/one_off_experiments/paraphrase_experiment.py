import json
from losses.triplet_loss import TripletLoss
from models.wrapped_sentence_transformer import WrappedSentenceTransformerModel
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from train_utils import set_seed
from training_loop_strategies.utils import (
    compute_dissimilarities,
    record_pick,
    select_next_correct,
)
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

    labels_mask = [False] * len(flat_questions)
    for index in relevant_sentence_indexes:
        labels_mask[index] = True

    flat_questions_paraphrased = item["context"]["questions_paraphrased"]
    paraphrased_vector = [True] * len(flat_questions_paraphrased)

    paraphrase_lut = {}
    for right, left in enumerate(relevant_sentence_indexes):
        right = right + len(flat_questions)
        paraphrase_lut[left] = right
        paraphrase_lut[right] = left

    all_questions = [*flat_questions, *flat_questions_paraphrased]
    all_labels_mask = [*labels_mask, *paraphrased_vector]

    all_labels_mask = torch.tensor(all_labels_mask, device="cuda:0")

    query = item["question"]

    left_labels_mask = all_labels_mask.clone()
    left_labels_mask[len(labels_mask) :] = 0

    right_labels_mask = all_labels_mask.clone()
    right_labels_mask[: len(labels_mask)] = 0

    assert left_labels_mask.sum() == right_labels_mask.sum()

    left_labels_mask = left_labels_mask.bool()
    right_labels_mask = right_labels_mask.bool()

    labels_masks = (left_labels_mask, right_labels_mask)

    return (
        query,
        all_questions,
        all_labels_mask,
        relevant_sentence_indexes,
        paraphrase_lut,
        labels_masks,
    )


def compute_inward_metric(criterion, similarities, labels_mask):
    metric = criterion(similarities, labels_mask)

    return metric


def compute_outward_metric(criterion, document_embeddings, labels_masks):
    left_labels_mask, right_labels_mask = labels_masks

    # Extract the relevant vectors
    left_vector = document_embeddings[left_labels_mask]
    right_vector = document_embeddings[right_labels_mask]

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
            all_labels_mask,
            relevant_sentence_indexes,
            paraphrase_lut,
            labels_masks,
        ) = load_paraphrased_row()

        query_embedding, document_embeddings = model.get_query_and_document_embeddings(
            query, all_questions
        )

        picked_mask = torch.zeros(len(all_questions), device="cuda:0", dtype=torch.bool)
        actual_selected_indices = []
        teacher_forcing = []
        all_inwards = []
        all_outwards = []
        all_labels_mask_list = [all_labels_mask.clone()]
        picked_mask_list = [picked_mask]

        recall_at_1 = 0

        original_all_labels_mask = all_labels_mask_list[0]

        num_correct_answers = len(relevant_sentence_indexes)
        can_be_picked_set = set(relevant_sentence_indexes)
        for _ in range(num_correct_answers):
            current_all_labels_mask = all_labels_mask_list[-1]
            current_picked_mask = picked_mask_list[-1]

            similarities = torch.matmul(
                document_embeddings, query_embedding.T
            ).squeeze()
            similarities = torch.clamp(similarities, min=0, max=1)

            inward = compute_inward_metric(
                criterion, similarities, original_all_labels_mask
            )
            all_inwards.append(inward)

            outward = compute_outward_metric(
                criterion, document_embeddings, labels_masks
            )
            all_outwards.append(outward)

            dissimilarities = compute_dissimilarities(
                document_embeddings, current_picked_mask, similarities
            )

            predictions = similarities * (1 - dissimilarities)
            labels = current_all_labels_mask.float()
            local_inward = criterion(predictions, labels)

            loss = torch.zeros([], device=local_inward.device)
            if enable_local_inward:
                loss = loss + local_inward

            if enable_inward:
                loss = loss + inward

            if enable_outward:
                loss = loss + outward

            cloned_predictions = predictions.clone()
            cloned_predictions[current_picked_mask] = 0
            selected_index = torch.argmax(cloned_predictions).item()
            actual_selected_indices.append(selected_index)

            next_correct, recall_at_1 = select_next_correct(
                similarities,
                paraphrase_lut,
                recall_at_1,
                can_be_picked_set,
                selected_index,
            )

            next_picked_mask, _ = record_pick(
                next_correct,
                can_be_picked_set,
                paraphrase_lut,
                current_all_labels_mask,
                all_labels_mask_list,
                current_picked_mask,
                picked_mask_list,
                teacher_forcing,
            )

            wandb.log(
                {
                    "loss": loss,
                    "local_inward": local_inward,
                    "inward": inward,
                    "outward": outward,
                    "recall_at_1": recall_at_1 / (next_picked_mask.sum().item()),
                }
            )

        # inward = torch.stack(all_inwards).mean()
        # outward = torch.stack(all_outwards).mean()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

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
