from pathlib import Path
import pandas as pd
from run_analysis_scripts.gain_histogram_utils import get_meta_and_data_dicts
import seaborn as sns
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

sns.set_theme()


def json_dir_to_df(base_path):
    base_path = Path(base_path)
    rows = []

    for file_name in tqdm(os.listdir(base_path)):
        meta, data = get_meta_and_data_dicts(base_path, file_name)

        if meta is None:
            continue

        if "score_cutoff_histogram" not in data:
            continue

        data_dict = data["score_cutoff_histogram"]
        score_histogram_resolution = data["gain_histogram_resolution"]

        for score, metrics_list in data_dict.items():
            for metrics in metrics_list:
                row = {
                    **meta,
                    **metrics,
                    "score": int(score) * score_histogram_resolution,
                }
                rows.append(row)

    df = pd.DataFrame(rows)

    return df


def plot_df(model_name, test_dataset_name, ddf, mean_df, idx_max_df):
    output_dir = Path("./artifacts/score_histogram")
    output_dir.mkdir(exist_ok=True, parents=True)

    sns.lineplot(data=ddf, x="score", y="f1", hue="experiment_name")
    plt.title(f"{model_name} - {test_dataset_name}: F1 by score")
    plt.xlabel("score")
    plt.ylabel("F1")
    plt.ylim(0, 1)

    # Draw lines for maximum F1 scores
    for experiment_name in ddf["experiment_name"].unique():
        # Ensure that max_df has a row for each experiment_name
        if experiment_name in idx_max_df.index:
            max_f1_idx = idx_max_df.loc[experiment_name, "f1"]
            score_at_max_f1 = mean_df.iloc[max_f1_idx]["score"]
            max_f1 = mean_df.iloc[max_f1_idx]["f1"]

            # Draw horizontal line at max F1 score
            plt.axhline(y=max_f1, color="gray", linestyle="--")

            # Draw vertical line at the score where max F1 was found
            plt.axvline(x=score_at_max_f1, ymax=max_f1, color="gray", linestyle="--")

            line = f" score: {score_at_max_f1:.2f}, F1: {max_f1:.2f}"
            print(" ", model_name, test_dataset_name, experiment_name, line)

            # Adding text for the F1 score. Adjust x and y positions as needed.
            plt.text(
                1e-3,
                max_f1,
                line,
                va="bottom",
                ha="left",
            )

    plt.savefig(output_dir / f"{test_dataset_name}_f1_{model_name}_by_score.png")
    plt.close()


if __name__ == "__main__":
    df = json_dir_to_df("artifacts/checkpoint_evals")
    # df.to_csv("./artifacts/score.csv", index=False)
    # df = pd.read_csv("./artifacts/score.csv")
    # print(df)

    for model_name in ["mpnet", "minilm"]:
        for test_dataset_name in ["hotpot_qa_with_q", "wiki_multihop_qa_with_q"]:
            ddf = df
            # ddf = ddf[ddf["score"] <= 0.6]
            ddf = ddf[ddf["model_name"] == model_name]
            ddf = ddf[ddf["test_dataset_name"] == test_dataset_name]
            ddf = ddf.reset_index()

            ddf = ddf[["experiment_name", "f1", "score"]]

            mean_df = ddf.groupby(["score", "experiment_name"]).mean().reset_index()
            idx_max_df = mean_df.groupby(["experiment_name"]).idxmax()

            plot_df(model_name, test_dataset_name, ddf, mean_df, idx_max_df)

            print("-" * 80)
