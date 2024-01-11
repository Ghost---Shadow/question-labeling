import argparse
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# python src/one_off_experiments/plot_gain_histograms.py --json_file=artifacts/checkpoint_evals/mpnet_hotpot_qa_with_q_a6b5a.json

# Define and parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--json_file", type=str, help="Path to the JSON file")
args = parser.parse_args()

# Load data from JSON file
with open(args.json_file, "r") as file:
    data = json.load(file)

experiment_name = data["config"]["architecture"]["semantic_search_model"][
    "real_checkpoint"
]
experiment_name = experiment_name.split("/")
experiment_name = experiment_name[min(2, len(experiment_name) - 1)]

# Ensure the output directory exists
output_dir = f"./artifacts/gain_histogram/{experiment_name}"
os.makedirs(output_dir, exist_ok=True)

# Extract data from 'gain_cutoff_histogram' key
data_dict = data["gain_cutoff_histogram"]
gain_histogram_resolution = data["gain_histogram_resolution"]

rows = []
for gain, metrics_list in data_dict.items():
    for metrics in metrics_list:
        row = {"gain": gain}
        row.update(metrics)
        rows.append(row)

df = pd.DataFrame(rows)
df["gain"] = pd.to_numeric(df["gain"])
df["gain"] = df["gain"] * gain_histogram_resolution

# Step 3: Create and save plots for each metric
metric_names = ["recall_at_6969", "precision_at_6969", "f1_at_6969", "real_k"]
name_map = {
    "recall_at_6969": "recall",
    "precision_at_6969": "precision",
    "f1_at_6969": "f1",
    "real_k": "k",
}
for metric in metric_names:
    # metric_df = df.groupby("gain")[metric].agg(["mean", "std"]).reset_index()
    # metric_df["std"] = metric_df["std"].fillna(0)

    plt.figure()

    # sns.barplot(metric_df, x="gain", y="mean", yerr=metric_df["std"])
    # sns.barplot(df, x="gain", y=metric)
    sns.lineplot(df, x="gain", y=metric)
    metric_name = name_map[metric]
    plt.title(f"{experiment_name}: {metric_name} by gain")
    plt.xlabel("Gain")
    plt.ylabel(metric_name)

    # Save the plot
    plt.savefig(f"{output_dir}/{metric_name}_plot.png")
    plt.close()
