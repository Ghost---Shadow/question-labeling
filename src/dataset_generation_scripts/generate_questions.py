from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor
from models.t5_model import T5ModelForQuestionGeneration
from models.openai_chat_model import OpenAIChatModel
import json
import os
from tqdm import tqdm


def add_question_to_row(model, row):
    def generate_question(sentence):
        return model.generate_question(sentence)

    all_questions = []

    # Create a single ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=15) as executor:
        # Store futures for each sentence in a dictionary to maintain order
        futures_dict = {}
        for paragraph_index, paragraph in enumerate(row["context"]["sentences"]):
            for sentence_index, sentence in enumerate(paragraph):
                future = executor.submit(generate_question, sentence)
                futures_dict[(paragraph_index, sentence_index)] = future

        # Organize the results into the structure of paragraphs and sentences
        for paragraph_index, paragraph in enumerate(row["context"]["sentences"]):
            paragraph_questions = []
            for sentence_index, _ in enumerate(paragraph):
                future = futures_dict[(paragraph_index, sentence_index)]
                paragraph_questions.append(future.result())
            all_questions.append(paragraph_questions)

    row["context"]["questions"] = all_questions
    return row


def convert_to_question_for_split(dataset, model, split, debug):
    split_path = f"./data/hotpotqa_with_qa_gpt35/{split}.jsonl"

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
                "name": "gpt-3.5-turbo",
                # "name": "t5",
                # "size": "base",
                # "device": "cuda:0",
            }
        }
    }
    # model = T5ModelForQuestionGeneration(config)
    model = OpenAIChatModel(config)
    convert_to_question_dataset(model, debug=False)
