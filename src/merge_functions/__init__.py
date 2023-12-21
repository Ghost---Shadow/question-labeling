from merge_functions.weighted_average_merge import WeightedAverageMerger
from merge_functions.dnn_merge import DnnMerger


class NoOpModel:
    def __init__(self, config):
        ...

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Specify a merge function in config")

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("Specify a merge function in config")

    def get_metrics(self):
        return {}

    def parameters(self):
        return []


MERGE_STRATEGY_LUT = {
    "weighted_average_merger": WeightedAverageMerger,
    "dnn_merger": DnnMerger,
    None: NoOpModel,
}
