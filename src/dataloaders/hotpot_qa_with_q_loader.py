from datasets import load_dataset
from torch.utils.data import DataLoader
from dataloaders import hotpot_qa_loader


def downstream_collate(batch, upstream_batch):
    batch_flat_questions = []
    batch_relevant_question_indexes = []
    batch_labels_mask = []
    batch_paraphrase_lut = []
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

        paraphrased_questions = []
        for paragraph_questions in item["context"]["paraphrased_questions"]:
            for question in paragraph_questions:
                paraphrased_questions.append(question)

        all_flat_question = [*flat_questions, *paraphrased_questions]
        batch_flat_questions.append(all_flat_question)

        # No paraphrase
        relevant_question_indexes = upstream_relevant_sentence_indexes
        batch_relevant_question_indexes.append(relevant_question_indexes)

        # Merge selection vector
        downstream_labels_mask = upstream_labels_mask
        labels_mask = [*upstream_labels_mask, *downstream_labels_mask]
        batch_labels_mask.append(labels_mask)

        # Paraphrase look up table
        offset = len(flat_questions)
        paraphrase_lut = {}
        for idx in relevant_question_indexes:
            paraphrase_idx = offset + idx

            paraphrase_lut[paraphrase_idx] = idx
            paraphrase_lut[idx] = paraphrase_idx
        batch_paraphrase_lut.append(paraphrase_lut)

    return {
        "flat_questions": batch_flat_questions,
        "relevant_question_indexes": batch_relevant_question_indexes,
        "labels_mask": batch_labels_mask,
        "paraphrase_lut": batch_paraphrase_lut,
    }


def collate_fn(batch):
    upstream_batch = hotpot_qa_loader.collate_fn(batch)
    downstream_batch = downstream_collate(batch, upstream_batch)

    return {
        **upstream_batch,
        **downstream_batch,
    }


def get_train_loader(batch_size):
    dataset = load_dataset("scholarly-shadows-syndicate/hotpotqa_with_qa_gpt35")
    # dataset = load_dataset("./data/hotpotqa_with_qa_gpt35")

    train_loader = DataLoader(
        dataset["train"],
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    return train_loader


def get_validation_loader(batch_size):
    dataset = load_dataset("scholarly-shadows-syndicate/hotpotqa_with_qa_gpt35")
    # dataset = load_dataset("./data/hotpotqa_with_qa_gpt35")

    val_loader = DataLoader(
        dataset["validation"],
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    return val_loader
