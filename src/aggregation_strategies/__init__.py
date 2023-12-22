from aggregation_strategies.submodular_mutual_information_strategy import (
    SubmodularMutualInformation,
)
from aggregation_strategies.weighted_average_strategy import WeightedEmbeddingAverage
from utils.noop import NoOpModule


AGGREGATION_STRATEGY_LUT = {
    "average": WeightedEmbeddingAverage,
    "smi": SubmodularMutualInformation,
    "noop": NoOpModule,
}
