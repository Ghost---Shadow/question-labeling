from datasets import load_dataset
from torch.utils.data import DataLoader
from dataloaders import wiki_multihop_qa_loader
from dataloaders import hotpot_qa_with_q_loader


def collate_fn(batch):
    upstream_batch = wiki_multihop_qa_loader.collate_fn(batch)
    downstream_batch = hotpot_qa_with_q_loader.downstream_collate(batch, upstream_batch)

    return {
        **upstream_batch,
        **downstream_batch,
    }


def get_train_loader(batch_size):
    dataset = load_dataset("scholarly-shadows-syndicate/2wikimultihopqa_with_q_gpt35")
    # dataset = load_dataset("./data/2wikimultihopqa_with_q_gpt35")

    train_loader = DataLoader(
        dataset["train"],
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    return train_loader


def get_validation_loader(batch_size):
    dataset = load_dataset("scholarly-shadows-syndicate/2wikimultihopqa_with_q_gpt35")
    # dataset = load_dataset("./data/2wikimultihopqa_with_q_gpt35")

    val_loader = DataLoader(
        dataset["validation"],
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    return val_loader
