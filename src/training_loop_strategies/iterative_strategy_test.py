import unittest
from dataloaders.hotpot_qa_with_q_loader import get_loader
from losses.mse_loss import MSELoss
from losses.kl_div_loss import KLDivLoss
import torch.optim as optim
from models.wrapped_sentence_transformer import WrappedSentenceTransformerModel
from training_loop_strategies.iterative_strategy import (
    eval_step,
    train_step,
    train_step_full_precision,
)
from aggregation_strategies.weighted_average_strategy import WeightedEmbeddingAverage
from aggregation_strategies.submodular_mutual_information_strategy import (
    SubmodularMutualInformation,
)


# python -m unittest training_loop_strategies.iterative_strategy_test.TestTrainStepMixedPrecision -v
# @unittest.skip("needs gpu")
class TestTrainStepMixedPrecision(unittest.TestCase):
    # python -m unittest training_loop_strategies.iterative_strategy_test.TestTrainStepMixedPrecision.test_mixed_precision -v
    def test_mixed_precision(self):
        config = {
            "architecture": {
                "semantic_search_model": {
                    "checkpoint": "all-mpnet-base-v2",
                    "device": "cuda:0",
                },
                "aggregation_strategy": {
                    "name": "average",
                    "merge_strategy": {"name": "weighted_average_merger"},
                },
            },
            "eval": {"k": [1, 2]},
        }
        wrapped_model = WrappedSentenceTransformerModel(config)
        optimizer = optim.AdamW(wrapped_model.model.parameters(), lr=1e-5)

        batch_size = 2

        train_loader, _ = get_loader(batch_size)

        batch = next(iter(train_loader))
        aggregation_model = WeightedEmbeddingAverage(config, wrapped_model)
        loss_fn = MSELoss()

        for _ in range(10):
            loss = train_step(
                config, wrapped_model, optimizer, batch, aggregation_model, loss_fn
            )
            print(loss)

    # python -m unittest training_loop_strategies.iterative_strategy_test.TestTrainStepMixedPrecision.test_mixed_precision_smi_kl_div -v
    def test_mixed_precision_smi_kl_div(self):
        config = {
            "architecture": {
                "semantic_search_model": {
                    "checkpoint": "all-mpnet-base-v2",
                    "device": "cuda:0",
                },
                "aggregation_strategy": {
                    "name": "smi",
                    "merge_strategy": {"name": "weighted_average_merger"},
                },
            },
            "eval": {"k": [1, 2]},
        }
        wrapped_model = WrappedSentenceTransformerModel(config)
        optimizer = optim.AdamW(wrapped_model.model.parameters(), lr=1e-5)

        batch_size = 2

        train_loader, _ = get_loader(batch_size)

        batch = next(iter(train_loader))
        aggregation_model = SubmodularMutualInformation(config, wrapped_model)
        loss_fn = KLDivLoss()

        for _ in range(10):
            loss = train_step(
                config, wrapped_model, optimizer, batch, aggregation_model, loss_fn
            )
            print(loss)


# python -m unittest training_loop_strategies.iterative_strategy_test.TestEvalStep -v
# @unittest.skip("needs gpu")
class TestEvalStep(unittest.TestCase):
    # python -m unittest training_loop_strategies.iterative_strategy_test.TestEvalStep.test_eval_step -v
    def test_eval_step(self):
        config = {
            "architecture": {
                "semantic_search_model": {
                    "checkpoint": "all-mpnet-base-v2",
                    "device": "cuda:0",
                },
                "aggregation_strategy": {
                    "name": "smi",
                    "merge_strategy": {"name": "weighted_average_merger"},
                },
            },
            "eval": {"k": [1, 2]},
        }
        wrapped_model = WrappedSentenceTransformerModel(config)

        batch_size = 2

        _, eval_loader = get_loader(batch_size)

        batch = next(iter(eval_loader))
        aggregation_fn = WeightedEmbeddingAverage(config, wrapped_model)
        loss_fn = MSELoss()

        loss = eval_step(config, wrapped_model, batch, aggregation_fn, loss_fn)

        print(loss)

    # python -m unittest training_loop_strategies.iterative_strategy_test.TestEvalStep.test_eval_step_kl_div_smi -v
    def test_eval_step_kl_div_smi(self):
        config = {
            "architecture": {
                "semantic_search_model": {
                    "checkpoint": "all-mpnet-base-v2",
                    "device": "cuda:0",
                },
                "aggregation_strategy": {
                    "name": "smi",
                    "merge_strategy": {"name": "weighted_average_merger"},
                },
            },
            "eval": {"k": [1, 2]},
        }
        wrapped_model = WrappedSentenceTransformerModel(config)

        batch_size = 2

        _, eval_loader = get_loader(batch_size)

        batch = next(iter(eval_loader))
        aggregation_fn = SubmodularMutualInformation(config, wrapped_model)
        loss_fn = KLDivLoss()

        loss = eval_step(config, wrapped_model, batch, aggregation_fn, loss_fn)

        print(loss)


# python -m unittest training_loop_strategies.iterative_strategy_test.TestTrainStep -v
# @unittest.skip("needs gpu")
class TestTrainStep(unittest.TestCase):
    # python -m unittest training_loop_strategies.iterative_strategy_test.TestTrainStep.test_full_precision -v
    def test_full_precision(self):
        config = {
            "architecture": {
                "semantic_search_model": {
                    "checkpoint": "all-mpnet-base-v2",
                    "device": "cuda:0",
                },
                "aggregation_strategy": {
                    "name": "weighted_average",
                },
            },
            "eval": {"k": [1, 2]},
        }
        wrapped_model = WrappedSentenceTransformerModel(config)
        optimizer = optim.AdamW(wrapped_model.model.parameters(), lr=1e-5)

        batch_size = 2

        train_loader, _ = get_loader(batch_size)

        batch = next(iter(train_loader))
        aggregation_fn = WeightedEmbeddingAverage(config, wrapped_model)
        loss_fn = MSELoss()

        for _ in range(10):
            loss = train_step_full_precision(
                config, wrapped_model, optimizer, batch, aggregation_fn, loss_fn
            )
            print(loss)

    # python -m unittest training_loop_strategies.iterative_strategy_test.TestTrainStep.test_full_precision_smi_kl_div -v
    def test_full_precision_smi_kl_div(self):
        config = {
            "architecture": {
                "semantic_search_model": {
                    "checkpoint": "all-mpnet-base-v2",
                    "device": "cuda:0",
                },
                "aggregation_strategy": {
                    "name": "smi",
                    "merge_strategy": {"name": "weighted_average_merger"},
                },
            },
            "eval": {"k": [1, 2]},
        }
        wrapped_model = WrappedSentenceTransformerModel(config)
        optimizer = optim.AdamW(wrapped_model.model.parameters(), lr=1e-5)

        batch_size = 2

        train_loader, _ = get_loader(batch_size)

        batch = next(iter(train_loader))
        aggregation_fn = SubmodularMutualInformation(config, wrapped_model)
        loss_fn = KLDivLoss()

        for _ in range(10):
            loss = train_step_full_precision(
                config, wrapped_model, optimizer, batch, aggregation_fn, loss_fn
            )
            print(loss)
