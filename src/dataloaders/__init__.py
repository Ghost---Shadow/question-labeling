from dataloaders import hotpot_qa_loader
from dataloaders import hotpot_qa_with_q_loader

DATA_LOADER_LUT = {
    "hotpot_qa": hotpot_qa_loader.get_loader,
    "hotpot_qa_with_q": hotpot_qa_with_q_loader,
}
