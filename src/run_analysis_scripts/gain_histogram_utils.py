import json

from pydash import get

SFT = "simple_finetuning"
QD = "quality_diversity"
CD = "cross_dataset"

EXPERIMENT_NAME_MAP = {
    "gpt35_mpnet_kldiv_qd": QD,
    "gpt35_mpnet_ni_kldiv": SFT,
    "gpt35_minilm_kldiv_qd": QD,
    "gpt35_minilm_kldiv_ni": SFT,
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

    experiment_name = EXPERIMENT_NAME_MAP[experiment_name]

    model_name = get(data, "config.architecture.semantic_search_model.name")
    assert model_name, model_name

    train_dataset_name = get(data, "config.datasets.train", None)
    test_dataset_name = data["dataset_name"]

    if train_dataset_name is not None and train_dataset_name != test_dataset_name:
        assert experiment_name != SFT, file_name
        experiment_name = CD

    meta = {
        "file_name": file_name,
        "experiment_name": experiment_name,
        "model_name": model_name,
        "train_dataset_name": train_dataset_name,
        "test_dataset_name": test_dataset_name,
    }

    return meta, data
