import json
from datasets import load_dataset
from tqdm import tqdm


def check_ids_in_jsonl(file_path):
    dataset = load_dataset("hotpot_qa", "distractor", split="validation")
    seen_rows = 0
    with open(file_path, "r") as f:
        for line, actual_row in tqdm(zip(f, dataset), total=7405):
            row = json.loads(line)
            assert actual_row["id"] == row["id"]
            seen_rows += 1

    assert seen_rows == len(dataset), seen_rows


file_path = "data/validation.jsonl"
check_ids_in_jsonl(file_path)
