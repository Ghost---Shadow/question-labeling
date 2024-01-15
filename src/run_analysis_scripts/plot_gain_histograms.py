import json
from pathlib import Path
import pandas as pd
from pydash import get
import seaborn as sns
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

sns.set_theme()

METRICS = ["recall", "precision", "f1", "real_k"]

EXPERIMENT_NAME_MAP = {
    "gpt35_mpnet_kldiv_qd": "quality_diversity",
    "gpt35_mpnet_ni_kldiv": "simple_finetuning",
    "baseline": "baseline",
}


def format_experiment_name(raw_name):
    if raw_name in EXPERIMENT_NAME_MAP:
        return EXPERIMENT_NAME_MAP[raw_name]

    return raw_name


def json_dir_to_df(base_path, wanted_test_dataset):
    base_path = Path(base_path)
    rows = []

    for file_name in tqdm(os.listdir(base_path)):
        if not file_name.endswith(".json"):
            continue

        file_path = base_path / file_name
        with open(file_path, "r") as file:
            data = json.load(file)

        experiment_name = get(data, "config.wandb.name", "baseline")

        if experiment_name not in EXPERIMENT_NAME_MAP:
            continue

        train_dataset_name = get(data, "config.datasets.train", None)
        test_dataset_name = data["dataset_name"]

        if wanted_test_dataset != test_dataset_name:
            continue

        if train_dataset_name is not None and train_dataset_name != test_dataset_name:
            experiment_name = "cross_dataset"

        # Extract data from 'gain_cutoff_histogram' key
        data_dict = data["gain_cutoff_histogram"]
        gain_histogram_resolution = data["gain_histogram_resolution"]

        for gain, metrics_list in data_dict.items():
            for metrics in metrics_list:
                row = {
                    **metrics,
                    "gain": int(gain) * gain_histogram_resolution,
                    "experiment": format_experiment_name(experiment_name),
                }
                row.update(metrics)
                rows.append(row)

    df = pd.DataFrame(rows)

    return df


def plot_df(df, max_df, idx_max_df, mean_df, test_dataset_name):
    output_dir = Path("./artifacts/gain_histogram")
    output_dir.mkdir(exist_ok=True, parents=True)

    for metric in tqdm(["f1"]):
        plt.figure()

        sns.lineplot(data=df, x="gain", y=metric, hue="experiment")
        plt.title(f"{metric} by gain")
        plt.xlabel("Gain")
        plt.ylabel(metric)
        if metric != "real_k":
            plt.ylim(0, 1)

        # Draw lines for maximum F1 scores
        for experiment in df["experiment"].unique():
            # Ensure that max_df has a row for each experiment
            if experiment in max_df.index:
                max_value = max_df.loc[experiment, metric]
                # Assuming max_df has a 'gain' column with the gain value for max F1
                max_f1_idx = idx_max_df.loc[experiment, "f1"]
                gain_at_max_f1 = mean_df.iloc[max_f1_idx]["gain"]

                # Draw horizontal line at max F1 score
                plt.axhline(y=max_value, color="gray", linestyle="--")

                # Draw vertical line at the gain where max F1 was found
                plt.axvline(
                    x=gain_at_max_f1, ymax=max_value, color="gray", linestyle="--"
                )

                line = f" Gain: {gain_at_max_f1:.2f}, F1: {max_value:.2f}"
                print(experiment, line)

                # Adding text for the F1 score. Adjust x and y positions as needed.
                plt.text(
                    1e-3,
                    max_value,
                    line,
                    va="bottom",
                    ha="left",
                )

        plt.savefig(output_dir / f"{test_dataset_name}_{metric}_by_gain.png")
        plt.close()


if __name__ == "__main__":
    for test_dataset_name in ["hotpot_qa_with_q", "wiki_multihop_qa_with_q"]:
        df = json_dir_to_df("artifacts/checkpoint_evals", test_dataset_name)
        df = df[df["gain"] <= 0.6]
        mean_df = df.groupby(["gain", "experiment"]).mean().reset_index()
        max_df = mean_df.groupby(["experiment"]).max()
        idx_max_df = mean_df.groupby(["experiment"]).idxmax()

        plot_df(df, max_df, idx_max_df, mean_df, test_dataset_name)
