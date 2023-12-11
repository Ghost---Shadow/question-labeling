import json
import os
from datasets import load_dataset
from models.openai_chat_model import OpenAIChatModel
from tqdm import tqdm
from models.t5_model import T5ModelForQuestionGeneration
from torch.utils.data import DataLoader, IterableDataset
from dataloaders import hotpot_qa_loader
from functools import lru_cache


@lru_cache(maxsize=1024)
def cached_generate_question(model, sentence):
    return model.generate_question(sentence)


def add_question_to_row(model, row):
    all_questions = []
    for paragraph in row["context"]["sentences"]:
        paragraph_questions = []
        for sentence in paragraph:
            question = cached_generate_question(model, sentence)
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


def collate_fn(batch):
    batch_flat_questions = []
    for item in batch:
        flat_questions = []
        for paragraph_questions in item["context"]["questions"]:
            for question in paragraph_questions:
                flat_questions.append(question)

        batch_flat_questions.append(flat_questions)

    return {
        **hotpot_qa_loader.collate_fn(batch),
        "flat_questions": batch_flat_questions,
    }


class CustomIterableDataset(IterableDataset):
    def __init__(self, file_path):
        self.file_path = file_path

    def parse_file(self):
        with open(self.file_path, "r") as f:
            for line in f:
                yield json.loads(line)

    def __iter__(self):
        return iter(self.parse_file())


def get_loader(batch_size):
    train_file_path = "./data/train.jsonl"
    validation_file_path = "./data/validation.jsonl"
    train_dataset = CustomIterableDataset(train_file_path)
    validation_dataset = CustomIterableDataset(validation_file_path)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,  # TODO
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader


if __name__ == "__main__":
    config = {
        "architecture": {
            "question_generator_model": {
                "name": "gpt-3.5-turbo",
                # "name": "t5",
                # "size": "base",
                # "device": "cuda:0",
            }
        }
    }
    # model = T5ModelForQuestionGeneration(config)
    model = OpenAIChatModel(config)
    convert_to_question_dataset(model, debug=True)
