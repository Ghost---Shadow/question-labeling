from merge_functions.weighted_average_merge import WeightedAverageMerger
from merge_functions.dnn_merge import DnnMerger
from utils.noop import NoOpModule


MERGE_STRATEGY_LUT = {
    "weighted_average_merger": WeightedAverageMerger,
    "dnn_merger": DnnMerger,
    None: NoOpModule,
}
