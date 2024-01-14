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


def json_dir_to_df(base_path):
    base_path = Path(base_path)
    rows = []

    for file_name in tqdm(os.listdir(base_path)):
        if not file_name.endswith(".json"):
            continue

        file_path = base_path / file_name
        with open(file_path, "r") as file:
            data = json.load(file)

        experiment_name = get(data, "config.wandb.name", "baseline")

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


def plot_df(df, max_df):
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

        # Draw horizontal lines for peak achievable gains
        for experiment in df["experiment"].unique():
            max_value = max_df.loc[experiment, metric]
            plt.axhline(y=max_value, color="gray", linestyle="--")
            # Adding text. Adjust the x position as needed.
            plt.text(0.1, max_value, f"{max_value:.2f}", va="bottom", ha="left")

        plt.savefig(output_dir / f"{metric}_by_gain.png")
        plt.close()


if __name__ == "__main__":
    df = json_dir_to_df("artifacts/checkpoint_evals")
    df = df[df["gain"] <= 0.6]
    # df.to_csv("./artifacts/gain_curves.csv")
    max_df = df.groupby(["gain", "experiment"]).mean().reset_index()
    max_df = max_df.groupby(["experiment"]).max()
    print(max_df)

    plot_df(df, max_df)
