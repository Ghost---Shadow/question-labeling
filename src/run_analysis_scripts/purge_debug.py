import os
import json
from tqdm import tqdm

directory_path = "artifacts/checkpoint_evals"

for filename in tqdm(os.listdir(directory_path)):
    if filename.endswith(".json"):
        file_path = os.path.join(directory_path, filename)

        with open(file_path, "r") as file:
            try:
                data = json.load(file)

                # Check if the file contains 'debug': true
                if data.get("debug") is True:
                    print(f"Deleting {filename} as it contains 'debug': true")
                    os.remove(file_path)

            except json.JSONDecodeError:
                print(f"Error reading JSON from {filename}, skipping file.")

            except Exception as e:
                print(f"An error occurred with file {filename}: {e}")
