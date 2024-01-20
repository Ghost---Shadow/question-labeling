import json

from pydash import get


EXPERIMENT_NAME_MAP = {
    "gpt35_mpnet_kldiv_qd": "quality_diversity",
    "gpt35_mpnet_ni_kldiv": "simple_finetuning",
    "gpt35_minilm_kldiv_qd": "quality_diversity",
    "gpt35_minilm_kldiv_ni": "simple_finetuning",
    "baseline": "baseline",
}


def get_meta_and_data_dicts(base_path, file_name):
    if not file_name.endswith(".json"):
        return None, None

    file_path = base_path / file_name
    with open(file_path, "r") as file:
        data = json.load(file)
    experiment_name = get(data, "config.wandb.name", "baseline")

    if experiment_name not in EXPERIMENT_NAME_MAP:
        return None, None

    model_name = get(data, "config.architecture.semantic_search_model.name")
    assert model_name, model_name

    train_dataset_name = get(data, "config.datasets.train", None)
    test_dataset_name = data["dataset_name"]

    if train_dataset_name is not None and train_dataset_name != test_dataset_name:
        experiment_name = "cross_dataset"

    meta = {
        "experiment_name": experiment_name,
        "model_name": model_name,
        "train_dataset_name": train_dataset_name,
        "test_dataset_name": test_dataset_name,
    }

    return meta, data
