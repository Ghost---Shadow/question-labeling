import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

directory = "./artifacts/knee_tensors"

data_for_plot = []

for value_type in tqdm(["similarities", "all_diversities", "all_predictions"]):
    for filename in os.listdir(directory):
        if filename.endswith(".npz"):
            # Extract the train step from the filename
            train_step = int(filename.split("_")[-1].split(".")[0])

            data = np.load(os.path.join(directory, filename))
            arr = data[value_type]

            # If the array is 1D, expand its dimensions
            if len(arr.shape) == 1:
                arr = np.expand_dims(arr, axis=0)

            for num_picked, row_arr in enumerate(arr):
                sorted_cumsum = np.cumsum(np.sort(row_arr)[::-1])
                for doc_idx, value in enumerate(sorted_cumsum):
                    data_for_plot.append(
                        (value_type, doc_idx, num_picked, train_step, value)
                    )

df = pd.DataFrame(
    data_for_plot,
    columns=["value_type", "doc_idx", "num_picked", "train_step", "value"],
)

df = df[df["train_step"] <= 10]
prediction_df = df[df["value_type"] == "all_predictions"]
diversity_df = df[df["value_type"] == "all_diversities"]
quality_df = df[df["value_type"] == "similarities"]

sns.set(style="whitegrid")
plt.figure(figsize=(12, 8))
sns.lineplot(
    data=diversity_df,
    x="doc_idx",
    y="value",
    hue="num_picked",
)
plt.title("Diversity vs Iteration Step")
plt.xlabel("Document Index")
plt.ylabel("Diversity")
plt.legend(title="Trajectory Index")
output_path = os.path.join(directory, "submodular_diversity_gain.png")
plt.savefig(output_path)
plt.close()

# Plot for diversity vs training step
sns.set(style="whitegrid")
plt.figure(figsize=(12, 8))
sns.lineplot(
    data=diversity_df,
    x="doc_idx",
    y="value",
    hue="train_step",
)
plt.title("Submodular Diversity Gain Curve")
plt.xlabel("Document Index")
plt.ylabel("Diversity")
output_path = os.path.join(directory, "submodular_diversity_over_time.png")
plt.savefig(output_path)
plt.close()

# Plot for prediction vs iteration step
sns.set(style="whitegrid")
plt.figure(figsize=(12, 8))
sns.lineplot(
    data=prediction_df,
    x="doc_idx",
    y="value",
    hue="num_picked",
)
plt.title("Quality-Diversity vs Documents Picked")
plt.xlabel("Document Index")
plt.ylabel("Quality-Diversity")
plt.legend(title="Trajectory Index")
output_path = os.path.join(directory, "submodular_qd_gain.png")
plt.savefig(output_path)
plt.close()

# Plot for prediction vs training step
sns.set(style="whitegrid")
plt.figure(figsize=(12, 8))
sns.lineplot(
    data=prediction_df,
    x="doc_idx",
    y="value",
    hue="train_step",
)
plt.title("Submodular Quality-Diversity Gain Curve")
plt.xlabel("Document Index")
plt.ylabel("Quality-Diversity")
output_path = os.path.join(directory, "submodular_qd_over_time.png")
plt.savefig(output_path)
plt.close()

sns.set(style="whitegrid")
plt.figure(figsize=(12, 8))
sns.lineplot(
    data=quality_df,
    x="doc_idx",
    y="value",
    hue="num_picked",
)
plt.title("Quality vs Documents Picked")
plt.xlabel("Document Index")
plt.ylabel("Quality")
plt.legend().remove()
output_path = os.path.join(directory, "submodular_quality_gain.png")
plt.savefig(output_path)
plt.close()

# Plot for quality vs training step
sns.set(style="whitegrid")
plt.figure(figsize=(12, 8))
sns.lineplot(
    data=quality_df,
    x="doc_idx",
    y="value",
    hue="train_step",
)
plt.title("Submodular Quality Gain Curve")
plt.xlabel("Document Index")
plt.ylabel("Quality")
output_path = os.path.join(directory, "submodular_quality_over_time.png")
plt.savefig(output_path)
plt.close()
