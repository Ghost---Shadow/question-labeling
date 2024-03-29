import unittest
from models.wrapped_deberta import WrappedDebertaModel
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler


# python -m unittest models.wrapped_deberta_test.TestWrappedDebertaModel -v
class TestWrappedDebertaModel(unittest.TestCase):
    # python -m unittest models.wrapped_deberta_test.TestWrappedDebertaModel.test_get_inner_products -v
    def test_get_inner_products(self):
        config = {
            "architecture": {
                "semantic_search_model": {
                    "checkpoint": "microsoft/deberta-v3-base",
                    # "device": "cuda:0",
                    "device": "cpu",
                }
            }
        }
        model = WrappedDebertaModel(config)
        query = "What color is the fruit that Alice loves?"
        documents = [
            "What fruit does Alice love?",
            "What fruit does Bob love?",
            "What is the color of apple?",
            "How heavy is an apple?",
        ]

        (
            question_embedding,
            document_embeddings,
        ) = model.get_query_and_document_embeddings(query, documents)
        inner_products = (question_embedding @ document_embeddings.T).squeeze()

        order = torch.argsort(inner_products, descending=True).cpu().numpy()
        actual = list(np.array(documents)[order])

        expected = [
            "How heavy is an apple?",
            "What is the color of apple?",
            "What fruit does Alice love?",
            "What fruit does Bob love?",
        ]

        assert actual == expected, actual


# python -m unittest models.wrapped_deberta_test.TestOverfit -v
# @unittest.skip("needs GPU")
class TestOverfit(unittest.TestCase):
    # python -m unittest models.wrapped_deberta_test.TestOverfit.test_overfit -v
    def test_overfit(self):
        config = {
            "architecture": {
                "semantic_search_model": {
                    "checkpoint": "microsoft/deberta-v3-base",
                    "device": "cuda:0",
                }
            }
        }
        wrapped_model = WrappedDebertaModel(config)
        query = "What color is the fruit that Alice loves?"
        documents = [
            "What fruit does Alice love?",
            "What fruit does Bob love?",
            "What is the color of apple?",
            "How heavy is an apple?",
        ]
        wrapped_model.model.train()
        target = torch.tensor([[1.0, 0.0, 1.0, 0.0]], device="cuda:0")

        optimizer = optim.AdamW(wrapped_model.get_all_trainable_parameters(), lr=1e-6)
        loss_fn = nn.MSELoss()

        # loss goes down
        for _ in range(100):
            # zero grad
            optimizer.zero_grad()

            (
                question_embedding,
                document_embeddings,
            ) = wrapped_model.get_query_and_document_embeddings(query, documents)
            inner_product = (question_embedding @ document_embeddings.T).squeeze()

            loss = loss_fn(inner_product.unsqueeze(0), target)
            print(loss.item())

            loss.backward()
            optimizer.step()

        print(inner_product)

    # python -m unittest models.wrapped_deberta_test.TestOverfit.test_overfit_amp -v
    def test_overfit_amp(self):
        config = {
            "architecture": {
                "semantic_search_model": {
                    "checkpoint": "microsoft/deberta-v3-base",
                    "device": "cuda:0",
                }
            }
        }
        wrapped_model = WrappedDebertaModel(config)
        query = "What color is the fruit that Alice loves?"
        documents = [
            "What fruit does Alice love?",
            "What fruit does Bob love?",
            "What is the color of apple?",
            "How heavy is an apple?",
        ]
        wrapped_model.model.train()
        target = torch.tensor([[1.0, 0.0, 1.0, 0.0]], device="cuda:0")

        optimizer = optim.AdamW(wrapped_model.get_all_trainable_parameters(), lr=1e-6)
        loss_fn = nn.MSELoss()
        scaler = GradScaler()

        for _ in range(100):
            optimizer.zero_grad()

            with autocast(dtype=torch.float16):
                (
                    question_embedding,
                    document_embeddings,
                ) = wrapped_model.get_query_and_document_embeddings(query, documents)
                inner_product = (question_embedding @ document_embeddings.T).squeeze()

                loss = loss_fn(inner_product.unsqueeze(0), target)
                print(loss.item())

            # Optimizer step
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        print(inner_product)
