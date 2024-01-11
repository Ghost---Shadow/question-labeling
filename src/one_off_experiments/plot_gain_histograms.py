import json
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

NAME_MAP = {
    "recall_at_6969": "recall",
    "precision_at_6969": "precision",
    "f1_at_6969": "f1",
    "real_k": "k",
}

EXPERIMENT_NAME_MAP = {
    "gpt35_mpnet_kldiv_qd": "quality_diversity",
    "gpt35_mpnet_kldiv_only_q": "only_q",
    "gpt35_mpnet_kldiv_only_d": "only_d",
    "gpt35_mpnet_ni_kldiv": "non_iterative",
}


def format_experiment_name(raw_name):
    for candidate_name, formatted_name in EXPERIMENT_NAME_MAP.items():
        if raw_name.startswith(candidate_name):
            return formatted_name

    return raw_name


def json_dir_to_df(base_path):
    base_path = Path(base_path)
    rows = []

    for file_name in tqdm(os.listdir(base_path)):
        if not file_name.endswith(".json"):
            continue

        file_path = base_path / file_name
        with open(file_path, "r") as file:
            data = json.load(file)

        experiment_name = data["config"]["architecture"]["semantic_search_model"][
            "real_checkpoint"
        ]
        experiment_name = experiment_name.split("/")
        experiment_name = experiment_name[min(2, len(experiment_name) - 1)]

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

    df.rename(columns=NAME_MAP, inplace=True)

    return df


def plot_df(df):
    # Ensure the output directory exists
    output_dir = Path("./artifacts/gain_histogram")
    output_dir.mkdir(exist_ok=True, parents=True)

    metric_names = list(NAME_MAP.values())
    for metric in tqdm(metric_names):
        plt.figure()

        sns.lineplot(df, x="gain", y=metric, hue="experiment")
        plt.title(f"{metric} by gain")
        plt.xlabel("Gain")
        plt.ylabel(metric)
        plt.ylim(0, 1)

        # Save the plot
        plt.savefig(output_dir / f"{metric}_plot.png")
        plt.close()


if __name__ == "__main__":
    df = json_dir_to_df("artifacts/checkpoint_evals")
    df = df[df["gain"] <= 0.5]
    plot_df(df)
    # max_df = df.groupby("experiment").max()
    # print(max_df)
