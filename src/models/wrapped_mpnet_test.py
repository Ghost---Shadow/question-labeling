import unittest
from models.wrapped_mpnet import WrappedMpnetModel
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler


# python -m unittest models.wrapped_mpnet_test.TestWrappedMpnetModel -v
class TestWrappedMpnetModel(unittest.TestCase):
    # python -m unittest models.wrapped_mpnet_test.TestWrappedMpnetModel.test_get_inner_products -v
    def test_get_inner_products(self):
        config = {
            "architecture": {
                "semantic_search_model": {
                    "checkpoint": "sentence-transformers/all-mpnet-base-v2",
                    "device": "cuda:0",
                    # "device": "cpu",
                }
            }
        }
        model = WrappedMpnetModel(config)
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
        inner_products = model.inner_product(question_embedding, document_embeddings)

        order = torch.argsort(inner_products, descending=True).cpu().numpy()
        actual = list(np.array(documents)[order])

        expected = [
            "What fruit does Alice love?",
            "What fruit does Bob love?",
            "What is the color of apple?",
            "How heavy is an apple?",
        ]

        assert actual == expected, actual

        assert str(question_embedding.device) == "cuda:0", question_embedding.device
        assert str(document_embeddings.device) == "cuda:0", document_embeddings.device
        assert str(inner_products.device) == "cuda:0", inner_products.device

    # python -m unittest models.wrapped_mpnet_test.TestWrappedMpnetModel.test_get_inner_products_streaming -v
    def test_get_inner_products_streaming(self):
        config = {
            "architecture": {
                "semantic_search_model": {
                    "checkpoint": "sentence-transformers/all-mpnet-base-v2",
                    "device": "cuda:0",
                    # "device": "cpu",
                }
            },
            "training": {
                "streaming": {
                    "batch_size": 2,
                }
            },
        }
        model = WrappedMpnetModel(config)
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
        ) = model.get_query_and_document_embeddings_streaming(query, documents)
        inner_products = model.inner_product_streaming(
            question_embedding, document_embeddings
        )

        order = torch.argsort(inner_products, descending=True).cpu().numpy()
        actual = list(np.array(documents)[order])

        expected = [
            "What fruit does Alice love?",
            "What fruit does Bob love?",
            "What is the color of apple?",
            "How heavy is an apple?",
        ]
        assert actual == expected, actual

        assert str(question_embedding.device) == "cuda:0", question_embedding.device
        assert str(document_embeddings.device) == "cpu", document_embeddings.device
        assert str(inner_products.device) == "cuda:0", inner_products.device

    # python -m unittest models.wrapped_mpnet_test.TestWrappedMpnetModel.test_get_inner_products_streaming_large_enough -v
    def test_get_inner_products_streaming_large_enough(self):
        config = {
            "architecture": {
                "semantic_search_model": {
                    "checkpoint": "sentence-transformers/all-mpnet-base-v2",
                    "device": "cuda:0",
                    # "device": "cpu",
                }
            },
            "training": {
                "streaming": {
                    "batch_size": 5,
                }
            },
        }
        model = WrappedMpnetModel(config)
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
        ) = model.get_query_and_document_embeddings_streaming(query, documents)
        inner_products = model.inner_product_streaming(
            question_embedding, document_embeddings
        )

        order = torch.argsort(inner_products, descending=True).cpu().numpy()
        actual = list(np.array(documents)[order])

        expected = [
            "What fruit does Alice love?",
            "What fruit does Bob love?",
            "What is the color of apple?",
            "How heavy is an apple?",
        ]

        assert actual == expected, actual

        assert str(question_embedding.device) == "cuda:0", question_embedding.device
        assert str(document_embeddings.device) == "cuda:0", document_embeddings.device
        assert str(inner_products.device) == "cuda:0", inner_products.device


# python -m unittest models.wrapped_mpnet_test.TestOverfit -v
# @unittest.skip("needs GPU")
class TestOverfit(unittest.TestCase):
    # python -m unittest models.wrapped_mpnet_test.TestOverfit.test_overfit -v
    def test_overfit(self):
        config = {
            "architecture": {
                "semantic_search_model": {
                    "checkpoint": "sentence-transformers/all-mpnet-base-v2",
                    "device": "cuda:0",
                }
            }
        }
        wrapped_model = WrappedMpnetModel(config)
        query = "What color is the fruit that Alice loves?"
        documents = [
            "What fruit does Alice love?",
            "What fruit does Bob love?",
            "What is the color of apple?",
            "How heavy is an apple?",
        ]
        wrapped_model.model.train()
        target = torch.tensor([[1.0, 0.0, 1.0, 0.0]], device="cuda:0")

        optimizer = optim.AdamW(wrapped_model.get_all_trainable_parameters(), lr=1e-5)
        loss_fn = nn.MSELoss()

        # loss goes down
        train_steps = 10
        for _ in range(train_steps):
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

    # python -m unittest models.wrapped_mpnet_test.TestOverfit.test_overfit_streaming -v
    def test_overfit_streaming(self):
        config = {
            "architecture": {
                "semantic_search_model": {
                    "checkpoint": "sentence-transformers/all-mpnet-base-v2",
                    "device": "cuda:0",
                }
            },
            "training": {
                "streaming": {
                    "batch_size": 2,
                }
            },
        }
        wrapped_model = WrappedMpnetModel(config)
        query = "What color is the fruit that Alice loves?"
        documents = [
            "What fruit does Alice love?",
            "What fruit does Bob love?",
            "What is the color of apple?",
            "How heavy is an apple?",
        ]
        wrapped_model.model.train()
        target = torch.tensor([[1.0, 0.0, 1.0, 0.0]], device="cuda:0")

        optimizer = optim.AdamW(wrapped_model.get_all_trainable_parameters(), lr=1e-5)
        loss_fn = nn.MSELoss()

        # loss goes down
        train_steps = 10
        for _ in range(train_steps):
            # zero grad
            optimizer.zero_grad()

            (
                question_embedding,
                document_embeddings,
            ) = wrapped_model.get_query_and_document_embeddings_streaming(
                query, documents
            )
            inner_product = wrapped_model.inner_product_streaming(
                question_embedding, document_embeddings
            )

            loss = loss_fn(inner_product.unsqueeze(0), target)
            print(loss.item())

            loss.backward()
            optimizer.step()

        print(inner_product)

    # python -m unittest models.wrapped_mpnet_test.TestOverfit.test_overfit_amp -v
    def test_overfit_amp(self):
        config = {
            "architecture": {
                "semantic_search_model": {
                    "checkpoint": "sentence-transformers/all-mpnet-base-v2",
                    "device": "cuda:0",
                }
            }
        }
        wrapped_model = WrappedMpnetModel(config)
        query = "What color is the fruit that Alice loves?"
        documents = [
            "What fruit does Alice love?",
            "What fruit does Bob love?",
            "What is the color of apple?",
            "How heavy is an apple?",
        ]
        wrapped_model.model.train()
        target = torch.tensor([[1.0, 0.0, 1.0, 0.0]], device="cuda:0")

        optimizer = optim.AdamW(wrapped_model.get_all_trainable_parameters(), lr=1e-5)
        loss_fn = nn.MSELoss()
        scaler = GradScaler()

        train_steps = 10
        for _ in range(train_steps):
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

    # python -m unittest models.wrapped_mpnet_test.TestOverfit.test_overfit_amp_streaming -v
    def test_overfit_amp_streaming(self):
        config = {
            "architecture": {
                "semantic_search_model": {
                    "checkpoint": "sentence-transformers/all-mpnet-base-v2",
                    "device": "cuda:0",
                }
            },
            "training": {
                "streaming": {
                    "batch_size": 2,
                }
            },
        }
        wrapped_model = WrappedMpnetModel(config)
        query = "What color is the fruit that Alice loves?"
        documents = [
            "What fruit does Alice love?",
            "What fruit does Bob love?",
            "What is the color of apple?",
            "How heavy is an apple?",
        ]
        wrapped_model.model.train()
        target = torch.tensor([[1.0, 0.0, 1.0, 0.0]], device="cuda:0")

        optimizer = optim.AdamW(wrapped_model.get_all_trainable_parameters(), lr=1e-5)
        loss_fn = nn.MSELoss()
        scaler = GradScaler()

        train_steps = 10
        start_loss = None
        end_loss = None
        for _ in range(train_steps):
            optimizer.zero_grad()

            with autocast(dtype=torch.float16):
                (
                    question_embedding,
                    document_embeddings,
                ) = wrapped_model.get_query_and_document_embeddings_streaming(
                    query, documents
                )
                inner_product = wrapped_model.inner_product_streaming(
                    question_embedding, document_embeddings
                )

                loss = loss_fn(inner_product.unsqueeze(0), target)
                end_loss = loss.item()
                # print(end_loss)

                if start_loss is None:
                    start_loss = end_loss

            # Optimizer step
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        assert start_loss > end_loss

    # python -m unittest models.wrapped_mpnet_test.TestOverfit.test_overfit_amp_streaming_large -v
    def test_overfit_amp_streaming_large(self):
        config = {
            "architecture": {
                "semantic_search_model": {
                    "checkpoint": "sentence-transformers/all-mpnet-base-v2",
                    "device": "cuda:0",
                }
            },
            "training": {
                "streaming": {
                    "batch_size": 5,
                }
            },
        }
        wrapped_model = WrappedMpnetModel(config)
        query = "What color is the fruit that Alice loves?"
        documents = [
            "What fruit does Alice love?",
            "What fruit does Bob love?",
            "What is the color of apple?",
            "How heavy is an apple?",
        ]
        wrapped_model.model.train()
        target = torch.tensor([[1.0, 0.0, 1.0, 0.0]], device="cuda:0")

        optimizer = optim.AdamW(wrapped_model.get_all_trainable_parameters(), lr=1e-5)
        loss_fn = nn.MSELoss()
        scaler = GradScaler()

        train_steps = 10
        start_loss = None
        end_loss = None
        for _ in range(train_steps):
            optimizer.zero_grad()

            with autocast(dtype=torch.float16):
                (
                    question_embedding,
                    document_embeddings,
                ) = wrapped_model.get_query_and_document_embeddings_streaming(
                    query, documents
                )
                inner_product = wrapped_model.inner_product_streaming(
                    question_embedding, document_embeddings
                )

                loss = loss_fn(inner_product.unsqueeze(0), target)
                end_loss = loss.item()
                # print(end_loss)

                if start_loss is None:
                    start_loss = end_loss

            # Optimizer step
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        assert start_loss > end_loss
