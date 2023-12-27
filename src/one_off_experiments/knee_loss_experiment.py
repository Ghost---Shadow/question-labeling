from losses.knee_loss import KneeLoss
from losses.triplet_loss import TripletLoss
from models.wrapped_sentence_transformer import WrappedSentenceTransformerModel
from one_off_experiments.paraphrase_experiment import load_paraphrased_row
from tqdm import tqdm
from train_utils import set_seed
import torch
from training_loop_strategies.utils import (
    compute_dissimilarities,
    record_pick,
    select_next_correct,
)
import wandb


def split_sort_merge(similarities, picked, paraphrase_lut):
    # Convert picked list to a tensor for efficient operations
    picked_tensor = torch.tensor(picked, device=similarities.device)

    alt_picked = [paraphrase_lut[p] for p in picked]
    alt_picked_tensor = torch.tensor(alt_picked, device=similarities.device)
    zero_mask = torch.ones_like(similarities)
    zero_mask[alt_picked_tensor] = 0
    similarities = similarities * zero_mask

    # Create masks
    picked_mask = torch.zeros_like(similarities, dtype=torch.bool)
    picked_mask[picked_tensor] = True
    not_picked_mask = ~picked_mask

    # Split the tensor into picked and not picked
    picked_similarities = similarities[picked_mask]
    not_picked_similarities = similarities[not_picked_mask]

    # Sort both tensors in descending order
    picked_similarities_sorted, _ = torch.sort(picked_similarities, descending=True)
    not_picked_similarities_sorted, _ = torch.sort(
        not_picked_similarities, descending=True
    )

    # Merge the sorted tensors
    merged_similarities = torch.cat(
        (picked_similarities_sorted, not_picked_similarities_sorted), dim=0
    )

    return merged_similarities


def train_session(seed, enable_local_inward, enable_knee):
    set_seed(seed)

    s = ""
    if enable_knee:
        s += "_k"
    if enable_local_inward:
        s += "_li"

    wandb.init(
        project="knee_experiment",
        name="knee_experiment" + s + f"_{seed}",
        config={
            "knee": enable_knee,
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
    optimizer = torch.optim.AdamW(model.model.parameters(), lr=1e-5)
    triplet_loss_fn = TripletLoss({})
    knee_loss_fn = KneeLoss({})

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
        device = model.model.device
        picked_mask = torch.zeros(len(all_questions), device=device, dtype=torch.bool)
        num_correct_answers = len(relevant_sentence_indexes)
        can_be_picked_set = set(relevant_sentence_indexes)
        all_selection_vector_list = [all_selection_vector.clone()]
        picked_mask_list = [picked_mask]
        actual_selected_indices = []
        teacher_forcing = []
        recall_at_1 = 0

        total_triplet_loss = torch.zeros([], device=device)

        similarities = torch.matmul(document_embeddings, query_embedding.T).squeeze()
        similarities = torch.clamp(similarities, min=0, max=1)

        for _ in range(num_correct_answers):
            current_all_selection_vector = all_selection_vector_list[-1]
            current_picked_mask = picked_mask_list[-1]

            dissimilarities = compute_dissimilarities(
                document_embeddings, current_picked_mask, similarities
            )

            predictions = similarities * (1 - dissimilarities)
            labels = current_all_selection_vector.float()
            triplet_loss = triplet_loss_fn(predictions, labels)
            total_triplet_loss += triplet_loss

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

            record_pick(
                next_correct,
                can_be_picked_set,
                paraphrase_lut,
                current_all_selection_vector,
                all_selection_vector_list,
                current_picked_mask,
                picked_mask_list,
                teacher_forcing,
            )

        total_triplet_loss = total_triplet_loss / num_correct_answers

        merged_similarities = split_sort_merge(
            similarities, teacher_forcing, paraphrase_lut
        )
        knee_loss = knee_loss_fn(merged_similarities, len(teacher_forcing))

        recall_at_1 = recall_at_1 / num_correct_answers

        loss = torch.zeros([], device=total_triplet_loss.device)
        if enable_local_inward:
            loss = loss + total_triplet_loss

        if enable_knee:
            loss = loss + knee_loss

        merged_similarities_cumsum = (
            merged_similarities.cumsum(0).detach().cpu().numpy()
        )
        plot = wandb.plot.line_series(
            xs=range(len(merged_similarities_cumsum)),
            ys=[merged_similarities_cumsum],
            keys=["Cumulative Sum"],
            title="Cumulative Sum of Merged Similarities",
            xname="Index",
        )

        wandb.log(
            {
                "loss": loss,
                "local_inward": total_triplet_loss.item(),
                "knee_loss": knee_loss.item(),
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
    seeds = [42, 43, 44]
    for seed in tqdm(seeds):
        for enable_local_inward, enable_knee in tqdm(permutations, leave=False):
            train_session(seed, enable_local_inward, enable_knee)
