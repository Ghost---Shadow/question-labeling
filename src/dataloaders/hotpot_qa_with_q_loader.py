from datasets import load_dataset
from torch.utils.data import DataLoader
from dataloaders import hotpot_qa_loader


def collate_fn(batch):
    batch_flat_questions = []
    for item in batch:
        flat_questions = []

        # TODO: Questions not computed yet
        sentences = item["context"]["sentences"]
        maybe_questions = item["context"].get("questions", sentences)
        for paragraph_questions in maybe_questions:
            for question in paragraph_questions:
                flat_questions.append(question)

        batch_flat_questions.append(flat_questions)

    return {
        **hotpot_qa_loader.collate_fn(batch),
        "flat_questions": batch_flat_questions,
    }


def get_loader(batch_size):
    validation_dataset = load_dataset("somebody-had-to-do-it/hotpotqa_with_qa_gpt35")

    # TODO: Questions not computed yet
    train_dataset = load_dataset("hotpot_qa", "distractor")

    train_loader = DataLoader(
        train_dataset["train"],
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        validation_dataset["validation"],
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader
