import unittest
from dataloaders.hotpot_qa_with_q_loader import get_train_loader, get_validation_loader
from losses.triplet_loss import TripletLoss
import torch.optim as optim
from models.wrapped_mpnet import WrappedMpnetModel
from training_loop_strategies.iterative_strategy import (
    eval_step,
    train_step,
)
from torch.cuda.amp import GradScaler


# python -m unittest training_loop_strategies.iterative_strategy_test.TestTrainStepMixedPrecision -v
# @unittest.skip("needs gpu")
class TestTrainStepMixedPrecision(unittest.TestCase):
    # python -m unittest training_loop_strategies.iterative_strategy_test.TestTrainStepMixedPrecision.test_mixed_precision -v
    def test_mixed_precision(self):
        config = {
            "architecture": {
                "semantic_search_model": {
                    "checkpoint": "sentence-transformers/all-mpnet-base-v2",
                    "device": "cuda:0",
                },
            },
            "eval": {
                "k": [5, 10],
                "disable_cutoff_gains": False,
            },
        }
        wrapped_model = WrappedMpnetModel(config)
        optimizer = optim.AdamW(wrapped_model.model.parameters(), lr=1e-5)

        batch_size = 2

        train_loader = get_train_loader(batch_size)

        batch = next(iter(train_loader))
        loss_fn = TripletLoss(config)
        scaler = GradScaler()

        for _ in range(10):
            metrics = train_step(
                config, scaler, wrapped_model, optimizer, batch, loss_fn
            )
            print(metrics)


# python -m unittest training_loop_strategies.iterative_strategy_test.TestEvalStep -v
# @unittest.skip("needs gpu")
class TestEvalStep(unittest.TestCase):
    # python -m unittest training_loop_strategies.iterative_strategy_test.TestEvalStep.test_eval_step -v
    def test_eval_step(self):
        config = {
            "architecture": {
                "semantic_search_model": {
                    "checkpoint": "sentence-transformers/all-mpnet-base-v2",
                    "device": "cuda:0",
                }
            },
            "eval": {
                "k": [5, 10],
                "disable_cutoff_gains": False,
            },
        }
        wrapped_model = WrappedMpnetModel(config)

        batch_size = 2

        eval_loader = get_validation_loader(batch_size)

        batch = next(iter(eval_loader))
        loss_fn = TripletLoss(config)
        scaler = None
        optimizer = None

        metrics = eval_step(config, scaler, wrapped_model, optimizer, batch, loss_fn)

        print(metrics)


if __name__ == "__main__":
    unittest.main()
