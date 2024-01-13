import wandb
from tqdm import tqdm
import pandas as pd

entity_name = "souradeep-nanda123"
project_name = "question_labeling_loss_fn_ablations"

api = wandb.Api()

runs = api.runs(f"{entity_name}/{project_name}")

dfs = []
for run in tqdm(runs):
    df = run.history()
    df["name"] = [run.name] * len(df)
    dfs.append(df)

all_df = pd.concat(dfs)

ALL_DF_PATH = "./artifacts/all_df.csv"

all_df.to_csv(ALL_DF_PATH)

all_df = pd.read_csv(ALL_DF_PATH)

all_df.groupby("name").max().to_json("./dump.json")
