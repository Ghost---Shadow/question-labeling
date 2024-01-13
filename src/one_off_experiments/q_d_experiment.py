from pathlib import Path
from losses.triplet_loss import TripletLoss
from models.wrapped_deberta import WrappedDebertaModel
from models.wrapped_mpnet import WrappedMpnetModel
from one_off_experiments.paraphrase_experiment import load_paraphrased_row
from tqdm import tqdm
from train_utils import set_seed
import torch
import wandb
import numpy as np
from torch.cuda.amp import GradScaler
import training_loop_strategies.iterative_strategy as iterative_strategy
import training_loop_strategies.non_iterative_strategy as non_iterative_strategy


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


def train_session(seed, enable_iterative, enable_quality, enable_diversity):
    set_seed(seed)

    s = ""
    if not enable_iterative:
        s += "_ni"
    if enable_quality:
        assert enable_iterative
        s += "_q"
    if enable_diversity:
        assert enable_iterative
        s += "_d"

    train_step = {
        True: iterative_strategy.train_step,
        False: non_iterative_strategy.train_step,
    }[enable_iterative]

    wandb.init(
        project="q_d_experiment",
        name="q_d_experiment" + s + f"_{seed}",
        # name="q_d_experiment_deberta" + s + f"_{seed}",
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
                # "checkpoint": "microsoft/deberta-v3-base",
                "device": "cuda:0",
            },
            "quality_diversity": {
                "disable_quality": not enable_quality,
                "disable_diversity": not enable_diversity,
            },
        },
        # "eval": {"k": [1, 5, 10]},
        "eval": {"k": [8]},  # len(relevant_sentence_indexes)
        "training": {
            "streaming": {
                "enabled": False,
                "batch_size": 64,
            }
        },
    }
    wrapped_model = WrappedMpnetModel(config)
    # wrapped_model = WrappedDebertaModel(config)
    optimizer = torch.optim.AdamW(wrapped_model.model.parameters(), lr=1e-5)
    loss_fn = TripletLoss(config)
    scaler = GradScaler()

    train_steps = 250
    for _ in tqdm(range(train_steps), leave=False):
        (
            question,
            flat_questions,
            labels_mask,
            relevant_sentence_indexes,
            paraphrase_lut,
        ) = load_paraphrased_row()

        batch = {
            "questions": [question],
            "flat_questions": [flat_questions],
            "labels_mask": [labels_mask],
            # "flat_questions": [
            #     [*flat_questions, *flat_questions, *flat_questions, *flat_questions]
            # ],
            # "labels_mask": [[*labels_mask, *labels_mask, *labels_mask, *labels_mask]],
            "relevant_sentence_indexes": [relevant_sentence_indexes],
            "paraphrase_lut": [paraphrase_lut],
        }
        metrics = train_step(config, scaler, wrapped_model, optimizer, batch, loss_fn)
        wandb.log(metrics)

    wandb.finish()


if __name__ == "__main__":
    permutations = [
        [True, True, True],  # QD
        [True, False, True],  # D
        [True, True, False],  # Q
        [False, False, False],  # NI
    ]
    # seeds = [42, 43, 44]
    seeds = [42]
    # seeds = [43, 44]
    for seed in tqdm(seeds):
        for enable_iterative, enable_quality, enable_diversity in tqdm(
            permutations, leave=False
        ):
            train_session(seed, enable_iterative, enable_quality, enable_diversity)
