from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor
from models.t5_model import T5ModelForQuestionGeneration
from models.openai_chat_model import OpenAIChatModel
import json
import os
from tqdm import tqdm


def add_paraphrased_question_to_row(model, row):
    # Already computed
    if "paraphrased_questions" in row["context"]:
        return row

    def generate_paraphrase(sentence):
        return model.generate_paraphrase(sentence)

    question_lut = {}
    for title, questions in zip(row["context"]["title"], row["context"]["questions"]):
        sent_counter = 0

        for question in questions:
            question_lut[(title, sent_counter)] = question
            sent_counter += 1

    paraphrased_questions = []
    with ThreadPoolExecutor(max_workers=100) as executor:
        futures = []
        for title, sent_id in zip(
            row["supporting_facts"]["title"], row["supporting_facts"]["sent_id"]
        ):
            key = (title, sent_id)
            if key not in question_lut:
                # There are many bad pointers
                continue
            question = question_lut[key]
            future = executor.submit(generate_paraphrase, question)
            futures.append(future)

        for future in futures:
            paraphrased_questions.append(future.result())

    row["context"]["paraphrased_questions"] = paraphrased_questions

    return row


def add_question_to_row(model, row):
    # Already computed
    if "questions" in row["context"]:
        return row

    def generate_question(sentence):
        return model.generate_question(sentence)

    all_questions = []

    # Create a single ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=100) as executor:
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
    old_split_path = f"./data/hotpotqa_with_qa_gpt35/{split}_old.jsonl"
    new_split_path = f"./data/hotpotqa_with_qa_gpt35/{split}.jsonl"

    # Load existing data from the new file
    processed_ids = set()
    if os.path.exists(new_split_path):
        with open(new_split_path, "r") as f:
            for line in f:
                row = json.loads(line)
                processed_ids.add(row["id"])  # Assuming each row has a unique 'id'

    # Load original data for reference
    original_data = {}
    if os.path.exists(old_split_path):
        with open(old_split_path, "r") as f:
            for line in f:
                row = json.loads(line)
                original_data[row["id"]] = row

    TRAIN_LIMIT = 15000

    # Process and append new rows
    with open(new_split_path, "a") as new_file:
        for current_row, row in enumerate(
            tqdm(dataset[split], total=min(TRAIN_LIMIT, len(dataset[split])))
        ):
            # Skip already processed rows
            if row["id"] in processed_ids:
                continue

            if debug and current_row >= 100:
                break

            # Limit for trainset for now
            if current_row >= TRAIN_LIMIT:
                break

            # Check and add missing components
            row = original_data.get(row["id"], row)
            if "question" not in row["context"]:
                row = add_question_to_row(model, row)
            if "paraphrased_question" not in row["context"]:
                row = add_paraphrased_question_to_row(model, row)

            new_file.write(json.dumps(row) + "\n")


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
