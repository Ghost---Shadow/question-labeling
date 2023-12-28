from dataloaders import hotpot_qa_loader
from dataloaders import hotpot_qa_with_q_loader
from dataloaders import wiki_multihop_qa_loader
from dataloaders import wiki_multihop_qa_with_q_loader


DATA_LOADER_LUT = {
    "hotpot_qa": (
        hotpot_qa_loader.get_train_loader,
        hotpot_qa_loader.get_validation_loader,
    ),
    "hotpot_qa_with_q": (
        hotpot_qa_with_q_loader.get_train_loader,
        hotpot_qa_with_q_loader.get_validation_loader,
    ),
    "wiki_multihop_qa": (
        wiki_multihop_qa_loader.get_train_loader,
        wiki_multihop_qa_loader.get_validation_loader,
    ),
    "wiki_multihop_qa_with_q": (
        wiki_multihop_qa_with_q_loader.get_train_loader,
        wiki_multihop_qa_with_q_loader.get_validation_loader,
    ),
}
