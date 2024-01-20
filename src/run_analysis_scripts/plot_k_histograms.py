import json
from pathlib import Path
import pandas as pd
from pydash import get
from run_analysis_scripts.plot_gain_histograms import EXPERIMENT_NAME_MAP
import seaborn as sns
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

sns.set_theme()


def json_dir_to_df(base_path):
    base_path = Path(base_path)
    rows = []

    for file_name in tqdm(os.listdir(base_path)):
        if not file_name.endswith(".json"):
            continue

        file_path = base_path / file_name
        with open(file_path, "r") as file:
            data = json.load(file)

        if data["debug"] is True:
            continue

        experiment_name = get(data, "config.wandb.name", "baseline")

        if experiment_name not in EXPERIMENT_NAME_MAP:
            # print(experiment_name)
            continue

        experiment_name = EXPERIMENT_NAME_MAP[experiment_name]

        model_name = get(data, "config.architecture.semantic_search_model.name")
        assert model_name, model_name

        if len(data["config"]["eval"]["k"]) == 3:
            continue

        train_dataset_name = get(data, "config.datasets.train", None)
        test_dataset_name = data["dataset_name"]

        if train_dataset_name is not None and train_dataset_name != test_dataset_name:
            experiment_name = "cross_dataset"

        # Extract data from 'k_cutoff_histogram' key
        metrics = data["metrics"]

        for i in range(1, 100):
            key = f"f1_at_{i}"

            if key not in metrics:
                continue

            row = {
                "k": i,
                "f1": metrics[key],
                "experiment": experiment_name,
                "train_dataset_name": train_dataset_name,
                "test_dataset_name": test_dataset_name,
                "model_name": model_name,
            }
            rows.append(row)

    df = pd.DataFrame(rows)

    return df


def plot_df(df, max_df, idx_max_df, test_dataset_name, model_name):
    output_dir = Path("./artifacts/gain_histogram")
    output_dir.mkdir(exist_ok=True, parents=True)

    plt.figure()
    metric = "f1"

    sns.lineplot(data=df, x="k", y=metric, hue="experiment")
    plt.title(f"{model_name} - {test_dataset_name}: {metric} by k")
    plt.xlabel("k")
    plt.ylabel(metric)
    plt.ylim(0, 1)

    # Draw lines for maximum F1 scores
    for experiment in tqdm(df["experiment"].unique()):
        # Ensure that max_df has a row for each experiment
        if experiment in max_df.index:
            max_value = max_df.loc[experiment, metric]
            # Assuming max_df has a 'k' column with the k value for max F1
            max_f1_idx = idx_max_df.loc[experiment, "f1"]
            k_at_max_f1 = df.iloc[max_f1_idx]["k"]

            # Draw horizontal line at max F1 score
            plt.axhline(y=max_value, color="gray", linestyle="--")

            # Draw vertical line at the k where max F1 was found
            plt.axvline(x=k_at_max_f1, ymax=max_value, color="gray", linestyle="--")

            line = f" k: {k_at_max_f1:.2f}, F1: {max_value:.2f}"
            print(" ", model_name, test_dataset_name, experiment, line)

            # Adding text for the F1 score. Adjust x and y positions as needed.
            plt.text(
                75,
                max_value,
                line,
                va="bottom",
                ha="left",
            )

    plt.savefig(output_dir / f"{test_dataset_name}_{metric}_{model_name}_by_k.png")
    plt.close()


if __name__ == "__main__":
    df = json_dir_to_df("artifacts/checkpoint_evals")
    # df.to_csv("./artifacts/temp.csv")
    # df = pd.read_csv("./artifacts/temp.csv")
    for model_name in ["mpnet", "minilm"]:
        for test_dataset_name in ["hotpot_qa_with_q", "wiki_multihop_qa_with_q"]:
            ddf = df
            ddf = ddf[ddf["test_dataset_name"] == test_dataset_name]
            ddf = ddf[ddf["model_name"] == model_name]
            ddf = ddf.reset_index()
            ddf = ddf[["k", "f1", "experiment"]]
            max_df = ddf.groupby(["experiment"]).max()
            idx_max_df = ddf.groupby(["experiment"]).idxmax()

            plot_df(ddf, max_df, idx_max_df, test_dataset_name, model_name)
