from datasets import load_dataset
from torch.utils.data import DataLoader


def collate_fn(batch):
    batch_questions = []
    batch_flat_sentences = []
    batch_relevant_sentence_indexes = []
    batch_selection_vector = []
    # batch_flag_for_error = []

    for item in batch:
        # flag_for_error = False
        question = item["question"]
        flat_sentences = []

        index_lut = {}
        for title, sentences in zip(
            item["context"]["title"], item["context"]["content"]
        ):
            sent_counter = 0
            for sentence in sentences:
                index_lut[(title, sent_counter)] = len(flat_sentences)
                flat_sentences.append(sentence)
                sent_counter += 1

        relevant_sentence_indexes = []
        for title, sent_id in zip(
            item["supporting_facts"]["title"], item["supporting_facts"]["sent_id"]
        ):
            key = (title, sent_id)
            if key in index_lut:
                flat_index = index_lut[key]
                relevant_sentence_indexes.append(flat_index)
            else:
                # flag_for_error = True
                # All rows should be clean
                assert False, item

        selection_vector = [False] * len(flat_sentences)
        for index in relevant_sentence_indexes:
            selection_vector[index] = True

        batch_questions.append(question)
        batch_flat_sentences.append(flat_sentences)
        batch_relevant_sentence_indexes.append(relevant_sentence_indexes)
        batch_selection_vector.append(selection_vector)
        # batch_flag_for_error.append(flag_for_error)

    return {
        "questions": batch_questions,
        "flat_sentences": batch_flat_sentences,
        "relevant_sentence_indexes": batch_relevant_sentence_indexes,
        "selection_vector": batch_selection_vector,
        # "flag_for_error": batch_flag_for_error,
    }


def get_train_loader(batch_size):
    dataset = load_dataset("somebody-had-to-do-it/2WikiMultihopQA")

    train_loader = DataLoader(
        dataset["train"],
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    return train_loader


def get_validation_loader(batch_size):
    dataset = load_dataset("somebody-had-to-do-it/2WikiMultihopQA")

    val_loader = DataLoader(
        dataset["dev"],
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    return val_loader
