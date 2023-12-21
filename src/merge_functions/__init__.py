from merge_functions.weighted_average_merge import WeightedAverageMerger
from merge_functions.dnn_merge import DnnMerger

MERGE_FUNCTIONS = {
    "weighted_average_merger": WeightedAverageMerger,
    "dnn_merger": DnnMerger,
}
