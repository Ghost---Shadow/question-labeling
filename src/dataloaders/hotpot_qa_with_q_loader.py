import json
import os
from datasets import load_dataset
from tqdm import tqdm
from models.t5_model import T5ModelForQuestionGeneration


def add_question_to_row(model, row):
    all_questions = []
    for paragraph in row["context"]["sentences"]:
        paragraph_questions = []
        for sentence in paragraph:
            question = model.generate_question(sentence)
            paragraph_questions.append(question)
        all_questions.append(paragraph_questions)

    row["context"]["questions"] = all_questions
    return row


def convert_to_question_for_split(dataset, model, split, debug):
    split_path = f"./data/{split}.jsonl"

    # How many rows has been computed so far
    done_so_far = 0
    if os.path.exists(split_path):
        with open(split_path) as f:
            for _ in f:
                done_so_far += 1

    current_row = -1
    for row in tqdm(dataset[split]):
        current_row += 1

        # Autoresume logic
        if current_row < done_so_far:
            continue

        if debug is True and current_row >= 100:
            break

        with open(split_path, "a") as f:
            modified_row = add_question_to_row(model, row)
            f.write(json.dumps(modified_row) + "\n")


def convert_to_question_dataset(model, debug=False):
    dataset = load_dataset("hotpot_qa", "distractor")

    convert_to_question_for_split(dataset, model, "train", debug)
    convert_to_question_for_split(dataset, model, "validation", debug)


if __name__ == "__main__":
    config = {
        "architecture": {
            "question_generator_model": {
                "name": "t5",
                "size": "base",
                "device": "cuda:0",
            }
        }
    }
    model = T5ModelForQuestionGeneration(config)
    convert_to_question_dataset(model, debug=True)
