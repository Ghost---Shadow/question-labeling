from datasets import load_dataset
from torch.utils.data import DataLoader
from dataloaders import hotpot_qa_loader


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


def get_loader(batch_size):
    dataset = load_dataset("somebody-had-to-do-it/hotpotqa_with_qa_gpt35")

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
