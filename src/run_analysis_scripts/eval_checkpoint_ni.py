import argparse
import json
from pathlib import Path
from models.checkpoint_manager import generate_md5_hash
import torch
from dataloaders import DATA_LOADER_LUT
from models import MODEL_LUT
from tqdm import tqdm
from training_loop_strategies.utils import (
    accumulate_gain_cutoff,
    average_metrics,
    compute_cutoff_gain,
    compute_cutoff_gain_histogram,
    compute_search_metrics,
)


def main(
    config,
    wrapped_model,
    validation_loader,
    gain_histogram_resolution,
    debug,
):
    all_metrics = []
    gain_histogram_accumulator = {}
    with torch.no_grad():
        pbar = tqdm(validation_loader)
        for batch in pbar:
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
                (
                    query_embedding,
                    document_embeddings,
                ) = wrapped_model.get_query_and_document_embeddings(
                    question, flat_questions
                )
                similarities = (query_embedding @ document_embeddings.T).squeeze()

                similarities = torch.clamp(similarities, min=0, max=1)
                predictions = similarities

                sorted_predictions, sorted_indices = predictions.sort(descending=True)
                ranking_predictions = sorted_predictions.tolist()
                ranking_indices = sorted_indices.tolist()

                cutoff_gain = compute_cutoff_gain(
                    ranking_indices,
                    ranking_predictions,
                    paraphrase_lut,
                    no_paraphrase_relevant_question_indexes,
                )

                search_metrics = compute_search_metrics(
                    config,
                    ranking_indices,
                    paraphrase_lut,
                    no_paraphrase_relevant_question_indexes,
                )

                cutoff_gain_histogram = compute_cutoff_gain_histogram(
                    ranking_indices,
                    ranking_predictions,
                    paraphrase_lut,
                    no_paraphrase_relevant_question_indexes,
                    resolution=gain_histogram_resolution,
                )

                accumulate_gain_cutoff(
                    gain_histogram_accumulator, cutoff_gain_histogram
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

    return avg_metrics, gain_histogram_accumulator


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to checkpoint. Type 'baseline' to ignore this",
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
    parser.add_argument(
        "--gain_histogram_resolution",
        type=float,
        required=False,
        default=1e-2,
        help="Resolution of gain histogram",
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    checkpoint_path = args.checkpoint_path
    model_type = args.model_type
    dataset_name = args.dataset_name
    device = args.device
    output_dir = args.output_dir
    gain_histogram_resolution = args.gain_histogram_resolution
    debug = args.debug

    base_hf_checkpoint = {
        "mpnet": "sentence-transformers/all-mpnet-base-v2",
    }[model_type]

    eval_config = {
        "architecture": {
            "semantic_search_model": {
                "name": model_type,
                "checkpoint": base_hf_checkpoint,
                "real_checkpoint": checkpoint_path,
                "device": device,
            }
        },
        "eval": {"k": list(range(1, 100))},
    }

    _, get_validation_loader = DATA_LOADER_LUT[dataset_name]
    validation_loader = get_validation_loader(batch_size=1)

    print("Loading model")

    wrapped_model = MODEL_LUT[model_type](eval_config)

    checkpoint = torch.load(checkpoint_path)
    wrapped_model.model.load_state_dict(checkpoint["model_state_dict"])
    config = checkpoint["config"]
    config["eval_train"] = config["eval"]
    config["eval"] = eval_config["eval"]
    assert config["training"]["strategy"]["name"] == "non_iterative_strategy"

    wrapped_model.model.eval()

    metrics, gain_cutoff_histogram = main(
        config,
        wrapped_model,
        validation_loader,
        gain_histogram_resolution,
        debug,
    )

    result = {
        "config": config,
        "metrics": metrics,
        "debug": debug,
        "dataset_name": dataset_name,
        "gain_cutoff_histogram": gain_cutoff_histogram,
        "gain_histogram_resolution": gain_histogram_resolution,
    }
    hashstr = generate_md5_hash(result)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    output_path = output_dir / f"{model_type}_{dataset_name}_{hashstr}.json"

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
