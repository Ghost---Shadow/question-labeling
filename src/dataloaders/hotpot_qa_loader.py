from datasets import load_dataset
from torch.utils.data import DataLoader

# from utils.decorators import dump_and_crash


# @dump_and_crash # Should be fixed now
def collate_fn(batch):
    batch_questions = []
    batch_flat_sentences = []
    batch_relevant_sentence_indexes = []
    batch_labels_mask = []
    # batch_flag_for_error = []

    for item in batch:
        # https://github.com/hotpotqa/hotpot/issues/47
        # flag_for_error = False
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
            key = (title, sent_id)
            if key in index_lut:
                flat_index = index_lut[key]
                relevant_sentence_indexes.append(flat_index)
            # else:
            #     flag_for_error = True

        # Sort if necessary
        relevant_sentence_indexes = sorted(relevant_sentence_indexes)

        labels_mask = [False] * len(flat_sentences)
        for index in relevant_sentence_indexes:
            labels_mask[index] = True

        batch_questions.append(question)
        batch_flat_sentences.append(flat_sentences)
        batch_relevant_sentence_indexes.append(relevant_sentence_indexes)
        batch_labels_mask.append(labels_mask)
        # batch_flag_for_error.append(flag_for_error)

    return {
        "questions": batch_questions,
        "flat_sentences": batch_flat_sentences,
        "relevant_sentence_indexes": batch_relevant_sentence_indexes,
        "labels_mask": batch_labels_mask,
        # "flag_for_error": batch_flag_for_error,
    }


def get_train_loader(batch_size):
    dataset = load_dataset("hotpot_qa", "distractor")

    train_loader = DataLoader(
        dataset["train"],
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    return train_loader


def get_validation_loader(batch_size):
    dataset = load_dataset("hotpot_qa", "distractor")

    val_loader = DataLoader(
        dataset["validation"],
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    return val_loader
