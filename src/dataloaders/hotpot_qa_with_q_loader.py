from datasets import load_dataset
from torch.utils.data import DataLoader
from dataloaders import hotpot_qa_loader


def collate_fn(batch):
    batch_flat_questions = []
    batch_relevant_question_indexes = []
    batch_labels_mask = []
    batch_paraphrase_lut = []

    upstream_batch = hotpot_qa_loader.collate_fn(batch)
    batch_upstream_relevant_sentence_indexes = upstream_batch[
        "relevant_sentence_indexes"
    ]
    batch_upstream_labels_mask = upstream_batch["labels_mask"]

    for item, upstream_relevant_sentence_indexes, upstream_labels_mask in zip(
        batch, batch_upstream_relevant_sentence_indexes, batch_upstream_labels_mask
    ):
        flat_questions = []

        for paragraph_questions in item["context"]["questions"]:
            for question in paragraph_questions:
                flat_questions.append(question)

        paraphrased_questions = item["context"]["paraphrased_questions"]
        all_flat_question = [*flat_questions, *paraphrased_questions]
        batch_flat_questions.append(all_flat_question)

        # Merge relevant_question_indexes
        downstream_relevant_question_indexes = [
            len(flat_questions) + i for i in range(len(paraphrased_questions))
        ]
        relevant_question_indexes = [
            *upstream_relevant_sentence_indexes,
            *downstream_relevant_question_indexes,
        ]
        batch_relevant_question_indexes.append(relevant_question_indexes)

        # Merge selection vector
        downstream_labels_mask = [True] * len(paraphrased_questions)
        labels_mask = [*upstream_labels_mask, *downstream_labels_mask]
        batch_labels_mask.append(labels_mask)

        # Paraphrase look up table
        paraphrase_lut = {}
        for left, right in zip(
            upstream_relevant_sentence_indexes, downstream_relevant_question_indexes
        ):
            paraphrase_lut[left] = right
            paraphrase_lut[right] = left
        batch_paraphrase_lut.append(paraphrase_lut)

    return {
        **upstream_batch,
        "flat_questions": batch_flat_questions,
        "relevant_question_indexes": batch_relevant_question_indexes,
        "labels_mask": batch_labels_mask,
        "paraphrase_lut": batch_paraphrase_lut,
    }


def get_train_loader(batch_size):
    dataset = load_dataset("somebody-had-to-do-it/hotpotqa_with_qa_gpt35")

    train_loader = DataLoader(
        dataset["train"],
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    return train_loader


def get_validation_loader(batch_size):
    dataset = load_dataset("somebody-had-to-do-it/hotpotqa_with_qa_gpt35")

    val_loader = DataLoader(
        dataset["validation"],
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    return val_loader
