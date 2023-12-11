import unittest
from dataloaders.hotpot_qa_with_q_loader import get_loader
from losses.masked_mse_loss import MaskedMSELoss
import torch.optim as optim
from models.wrapped_sentence_transformer import WrappedSentenceTransformerModel
from training_loop_strategies.iterative_strategy import train_step
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
            loss = train_step(wrapped_model, optimizer, batch, aggregation_fn, loss_fn)
            print(loss)
