import argparse
import json
from pathlib import Path
from models.checkpoint_manager import generate_md5_hash
import torch
from dataloaders import DATA_LOADER_LUT
from models import MODEL_LUT
from tqdm import tqdm
from training_loop_strategies.utils import (
    average_metrics,
    compute_cutoff_gain,
    compute_search_metrics,
    rerank_documents,
)


def main(wrapped_model, validation_loader, debug):
    all_metrics = []
    with torch.no_grad():
        for batch in tqdm(validation_loader):
            for (
                question,
                flat_questions,
                _,
                no_paraphrase_relevant_question_indexes,
                paraphrase_lut,
            ) in zip(
                batch["questions"],
                batch["flat_questions"],
                batch["labels_mask"],
                batch["relevant_sentence_indexes"],
                batch["paraphrase_lut"],
            ):
                actual_picks, actual_pick_prediction = rerank_documents(
                    wrapped_model,
                    question,
                    flat_questions,
                )

                cutoff_gain = compute_cutoff_gain(
                    actual_picks,
                    actual_pick_prediction,
                    paraphrase_lut,
                    no_paraphrase_relevant_question_indexes,
                )

                search_metrics = compute_search_metrics(
                    config,
                    actual_picks,
                    paraphrase_lut,
                    no_paraphrase_relevant_question_indexes,
                )

                all_metrics.append(
                    {
                        **search_metrics,
                        "cutoff_gain": cutoff_gain,
                    }
                )

            if debug and len(all_metrics) > 5:
                break

    avg_metrics = average_metrics(all_metrics)

    return avg_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_path", type=str, required=True, help="Path to checkpoint"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        help="Type of model e.g. mpnet or deberta",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Type of dataset e.g. hotpot_qa_with_q or wiki_multihop_qa_with_q",
    )
    parser.add_argument(
        "--device",
        type=str,
        required=False,
        default="cuda:0",
        help="Device to load search model e.g. cuda:0 or cpu",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=False,
        default="./artifacts/checkpoint_evals",
        help="Device to load search model e.g. cuda:0 or cpu",
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    checkpoint_path = args.checkpoint_path
    model_type = args.model_type
    dataset_name = args.dataset_name
    device = args.device
    output_dir = args.output_dir
    debug = args.debug

    config = {
        "architecture": {
            "semantic_search_model": {
                "name": model_type,
                "checkpoint": checkpoint_path,
                "device": device,
            }
        },
        "eval": {"k": [1, 5, 10]},
    }

    _, get_validation_loader = DATA_LOADER_LUT[dataset_name]
    validation_loader = get_validation_loader(batch_size=1)

    wrapped_model = MODEL_LUT[model_type](config)
    wrapped_model.model.eval()

    metrics = main(wrapped_model, validation_loader, debug)

    result = {
        "config": config,
        "metrics": metrics,
        "debug": debug,
        "dataset_name": dataset_name,
    }
    hashstr = generate_md5_hash(result)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    output_path = output_dir / f"{model_type}_{dataset_name}_{hashstr}.json"

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
