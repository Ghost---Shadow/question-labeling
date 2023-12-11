import unittest
from dataloaders.hotpot_qa_with_q_loader import get_loader
from losses.masked_mse_loss import MaskedMSELoss
import torch.optim as optim
from models.wrapped_sentence_transformer import WrappedSentenceTransformerModel
from training_loop_strategies.iterative_strategy import (
    eval_step,
    train_step,
    train_step_full_precision,
)
from aggregation_strategies.weighted_average_strategy import weighted_embedding_average


# python -m unittest training_loop_strategies.iterative_strategy_test.TestTrainStep -v
@unittest.skip("needs gpu")
class TestTrainStep(unittest.TestCase):
    # python -m unittest training_loop_strategies.iterative_strategy_test.TestTrainStep.test_full_precision -v
    def test_full_precision(self):
        config = {
            "architecture": {
                "semantic_search_model": {
                    "checkpoint": "all-mpnet-base-v2",
                    "device": "cuda:0",
                }
            }
        }
        wrapped_model = WrappedSentenceTransformerModel(config)
        optimizer = optim.AdamW(wrapped_model.model.parameters(), lr=1e-5)

        batch_size = 2

        train_loader, _ = get_loader(batch_size)

        batch = next(iter(train_loader))
        aggregation_fn = weighted_embedding_average
        loss_fn = MaskedMSELoss()

        for _ in range(10):
            loss = train_step_full_precision(
                wrapped_model, optimizer, batch, aggregation_fn, loss_fn
            )
            print(loss)


# python -m unittest training_loop_strategies.iterative_strategy_test.TestEvalStep -v
@unittest.skip("needs gpu")
class TestEvalStep(unittest.TestCase):
    # python -m unittest training_loop_strategies.iterative_strategy_test.TestEvalStep.test_eval_step -v
    def test_eval_step(self):
        config = {
            "architecture": {
                "semantic_search_model": {
                    "checkpoint": "all-mpnet-base-v2",
                    "device": "cuda:0",
                }
            }
        }
        wrapped_model = WrappedSentenceTransformerModel(config)

        batch_size = 2

        _, eval_loader = get_loader(batch_size)

        batch = next(iter(eval_loader))
        aggregation_fn = weighted_embedding_average
        loss_fn = MaskedMSELoss()

        avg_loss, avg_recall_at_k = eval_step(
            wrapped_model, batch, aggregation_fn, loss_fn
        )

        print(f"Avg Loss: {avg_loss}, Avg Recall@K: {avg_recall_at_k}")


# python -m unittest training_loop_strategies.iterative_strategy_test.TestTrainStep -v
# @unittest.skip("needs gpu")
class TestTrainStep(unittest.TestCase):
    # python -m unittest training_loop_strategies.iterative_strategy_test.TestTrainStep.test_mixed_precision -v
    def test_mixed_precision(self):
        config = {
            "architecture": {
                "semantic_search_model": {
                    "checkpoint": "all-mpnet-base-v2",
                    "device": "cuda:0",
                }
            }
        }
        wrapped_model = WrappedSentenceTransformerModel(config)
        optimizer = optim.AdamW(wrapped_model.model.parameters(), lr=1e-5)

        batch_size = 2

        train_loader, _ = get_loader(batch_size)

        batch = next(iter(train_loader))
        aggregation_fn = weighted_embedding_average
        loss_fn = MaskedMSELoss()

        for _ in range(10):
            loss = train_step(wrapped_model, optimizer, batch, aggregation_fn, loss_fn)
            print(loss)
