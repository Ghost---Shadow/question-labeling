from datasets import load_dataset
from torch.utils.data import DataLoader
from dataloaders import wiki_multihop_qa_loader


def collate_fn(batch):
    batch_flat_questions = []
    batch_relevant_question_indexes = []
    batch_selection_vector = []
    batch_paraphrase_lut = []

    upstream_batch = wiki_multihop_qa_loader.collate_fn(batch)
    batch_upstream_relevant_sentence_indexes = upstream_batch[
        "relevant_sentence_indexes"
    ]
    batch_upstream_selection_vector = upstream_batch["selection_vector"]

    for item, upstream_relevant_sentence_indexes, upstream_selection_vector in zip(
        batch, batch_upstream_relevant_sentence_indexes, batch_upstream_selection_vector
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
        downstream_selection_vector = [1] * len(paraphrased_questions)
        selection_vector = [*upstream_selection_vector, *downstream_selection_vector]
        batch_selection_vector.append(selection_vector)

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
        "selection_vector": batch_selection_vector,
        "paraphrase_lut": batch_paraphrase_lut,
    }


def get_loader(batch_size):
    dataset = load_dataset("somebody-had-to-do-it/2wikimultihopqa_with_q_gpt35")

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
