import json
import pandas as pd
import os

from pydash import get

# Path to the directory containing JSON files
directory_path = "artifacts/checkpoint_evals"

# Initialize a list to store the data
data_list = []

# Loop through each file in the directory
for filename in os.listdir(directory_path):
    if filename.endswith(".json"):
        file_path = os.path.join(directory_path, filename)

        # Open and load the JSON file
        with open(file_path, "r") as file:
            data = json.load(file)
            metrics = data["metrics"]

            # Extract the name under data['config']['wandb']
            name = get(data, "config.wandb.name", "baseline")

            # Add the metrics and name to the list
            metrics["name"] = name
            data_list.append(metrics)

# Convert the list to a DataFrame
df = pd.DataFrame(data_list)

# print(df[["name", "f1_at_1", "f1_at_5", "f1_at_10"]].sort_values(by="f1_at_1"))
print(df[["name", "f1_at_1", "f1_at_5", "f1_at_10"]].sort_values(by="f1_at_5").round(4))
# print(df[["name", "f1_at_1", "f1_at_5", "f1_at_10"]].sort_values(by="f1_at_10"))
