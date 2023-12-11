from datasets import load_dataset
from torch.utils.data import DataLoader


def collate_fn(batch):
    batch_questions = []
    batch_flat_sentences = []
    batch_relevant_sentence_indexes = []
    batch_selection_vector = []

    for item in batch:
        question = item["question"]
        flat_sentences = []

        index_lut = {}
        for title, sentences in zip(
            item["context"]["title"], item["context"]["sentences"]
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
            flat_index = index_lut[(title, sent_id)]
            relevant_sentence_indexes.append(flat_index)

        selection_vector = [0] * len(flat_sentences)
        for index in relevant_sentence_indexes:
            selection_vector[index] = 1

        batch_questions.append(question)
        batch_flat_sentences.append(flat_sentences)
        batch_relevant_sentence_indexes.append(relevant_sentence_indexes)
        batch_selection_vector.append(selection_vector)

    return {
        "questions": batch_questions,
        "flat_sentences": batch_flat_sentences,
        "relevant_sentence_indexes": batch_relevant_sentence_indexes,
        "selection_vector": batch_selection_vector,
    }


def get_loader(batch_size):
    dataset = load_dataset("hotpot_qa", "distractor")

    train_loader = DataLoader(
        dataset["train"],
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        dataset["validation"],
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader
