import unittest
from models.wrapped_sentence_transformer import WrappedSentenceTransformerModel
import numpy as np
import torch


# python -m unittest models.wrapped_sentence_transformer_test.TestWrappedSentenceTransformerModel -v
# @unittest.skip("needs GPU")
class TestWrappedSentenceTransformerModel(unittest.TestCase):
    # python -m unittest models.wrapped_sentence_transformer_test.TestWrappedSentenceTransformerModel.test_get_inner_products -v
    def test_get_inner_products(self):
        config = {
            "architecture": {
                "semantic_search_model": {
                    "checkpoint": "all-mpnet-base-v2",
                    "device": "cuda:0",
                }
            }
        }
        model = WrappedSentenceTransformerModel(config)
        query = "What color is the fruit that Alice loves?"
        documents = [
            "What fruit does Alice love?",
            "What fruit does Bob love?",
            "What is the color of apple?",
            "How heavy is an apple?",
        ]

        inner_products = model.get_inner_products(query, documents)
        order = torch.argsort(inner_products, descending=True).cpu().numpy()
        actual = list(np.array(documents)[order])

        expected = [
            "What fruit does Alice love?",
            "What fruit does Bob love?",
            "What is the color of apple?",
            "How heavy is an apple?",
        ]

        assert actual == expected, actual
