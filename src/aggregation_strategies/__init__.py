from aggregation_strategies.submodular_mutual_information_strategy import (
    submodular_mutual_information,
)
from aggregation_strategies.weighted_average_strategy import weighted_embedding_average


AGGREGATION_STRATEGY_LUT = {
    "average": weighted_embedding_average,
    "smi": submodular_mutual_information,
}
