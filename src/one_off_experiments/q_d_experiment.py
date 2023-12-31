from pathlib import Path
from losses.triplet_loss import TripletLoss
from models.wrapped_sentence_transformer import WrappedSentenceTransformerModel
from one_off_experiments.paraphrase_experiment import load_paraphrased_row
from tqdm import tqdm
from train_utils import set_seed
import torch
from training_loop_strategies.utils import (
    compute_cutoff_gain,
    compute_dissimilarities,
    record_pick,
    select_next_correct,
)
import wandb
import numpy as np


def gpu_to_numpy(tensor):
    return tensor.clone().detach().cpu().numpy()


def checkpoint_knee_tensor(train_steps, all_diversities, all_predictions, similarities):
    knee_tensor = {
        "similarities": gpu_to_numpy(similarities),
        "all_diversities": [gpu_to_numpy(diversity) for diversity in all_diversities],
        "all_predictions": [gpu_to_numpy(prediction) for prediction in all_predictions],
    }
    base_path = Path("./artifacts/knee_tensors")
    base_path.mkdir(exist_ok=True, parents=True)
    np.savez(base_path / f"step_{train_steps}.npz", **knee_tensor)


def train_session(seed, enable_quality, enable_diversity):
    set_seed(seed)

    s = ""
    if enable_quality:
        s += "_q"
    if enable_diversity:
        s += "_d"

    wandb.init(
        project="q_d_experiment",
        name="q_d_experiment" + s + f"_{seed}",
        config={
            "quality": enable_quality,
            "diversity": enable_diversity,
            "seed": seed,
        },
    )

    config = {
        "architecture": {
            "semantic_search_model": {
                "checkpoint": "sentence-transformers/all-mpnet-base-v2",
                "device": "cuda:0",
            }
        }
    }
    model = WrappedSentenceTransformerModel(config)
    optimizer = torch.optim.AdamW(model.model.parameters(), lr=1e-5)
    triplet_loss_fn = TripletLoss({})

    train_steps = 250
    for _ in tqdm(range(train_steps), leave=False):
        (
            query,
            all_questions,
            labels_mask,
            relevant_sentence_indexes,
            paraphrase_lut,
            labels_masks,
        ) = load_paraphrased_row()

        query_embedding, document_embeddings = model.get_query_and_document_embeddings(
            query, all_questions
        )
        device = model.model.device
        picked_mask = torch.zeros(len(all_questions), device=device, dtype=torch.bool)
        num_correct_answers = len(relevant_sentence_indexes)
        can_be_picked_set = set(relevant_sentence_indexes)
        labels_mask_list = [labels_mask.clone()]
        picked_mask_list = [picked_mask]
        actual_selected_indices = []
        teacher_forcing = []
        all_diversities = []
        all_predictions = []
        recall_at_1 = 0
        total_cutoff_gain = 0

        total_triplet_loss = torch.zeros([], device=device)

        similarities = torch.matmul(document_embeddings, query_embedding.T).squeeze()
        similarities = torch.clamp(similarities, min=0, max=1)

        for _ in range(num_correct_answers):
            current_labels_mask = labels_mask_list[-1]
            current_picked_mask = picked_mask_list[-1]

            dissimilarities = compute_dissimilarities(
                document_embeddings, current_picked_mask, similarities
            )

            predictions = torch.ones_like(similarities, device=device)
            if enable_quality:
                predictions = predictions * similarities
            if enable_diversity:
                predictions = predictions * (1 - dissimilarities)

            # all_diversities.append(1 - dissimilarities)
            # all_predictions.append(predictions)

            labels = current_labels_mask.float()
            triplet_loss = triplet_loss_fn(predictions, labels)
            total_triplet_loss += triplet_loss

            cutoff_gain = compute_cutoff_gain(
                predictions,
                labels_mask_list[0].clone(),
                current_picked_mask,
                paraphrase_lut,
            )
            if cutoff_gain is not None:
                total_cutoff_gain += cutoff_gain

            cloned_predictions = predictions.clone()
            cloned_predictions[current_picked_mask] = 0
            selected_index = torch.argmax(cloned_predictions).item()
            actual_selected_indices.append(selected_index)

            next_correct = select_next_correct(
                predictions, paraphrase_lut, can_be_picked_set, current_picked_mask
            )

            record_pick(
                next_correct,
                can_be_picked_set,
                paraphrase_lut,
                labels_mask_list,
                picked_mask_list,
                teacher_forcing,
            )

        # checkpoint_knee_tensor(
        #     train_step, all_diversities, all_predictions, similarities
        # )

        total_triplet_loss = total_triplet_loss / num_correct_answers

        recall_at_1 = recall_at_1 / num_correct_answers

        loss = total_triplet_loss

        similarities_cumsum = (
            similarities.sort(descending=True)[0].cumsum(0).detach().cpu().numpy()
        )
        plot = wandb.plot.line_series(
            xs=range(len(similarities_cumsum)),
            ys=[similarities_cumsum],
            keys=["Cumulative Sum"],
            title="Cumulative Sum of Merged Similarities",
            xname="Index",
        )

        wandb.log(
            {
                "loss": loss,
                "cutoff_gain": total_cutoff_gain / num_correct_answers,
                "local_inward": total_triplet_loss.item(),
                "recall_at_1": recall_at_1,
                "cumulative_sum": plot,
            }
        )

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    wandb.finish()


if __name__ == "__main__":
    permutations = [
        [True, True],
        [False, True],
        [True, False],
    ]
    # seeds = [42, 43, 44]
    seeds = [42]
    for seed in tqdm(seeds):
        for enable_quality, enable_diversity in tqdm(permutations, leave=False):
            train_session(seed, enable_quality, enable_diversity)
