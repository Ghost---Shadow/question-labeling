from aggregation_strategies.submodular_mutual_information_strategy import (
    SubmodularMutualInformation,
)
from models.wrapped_sentence_transformer import WrappedSentenceTransformerModel
import torch
import unittest


# python -m unittest aggregation_strategies.submodular_mutual_information_strategy_test.TestSubmodularMutualInformation -v
class TestSubmodularMutualInformation(unittest.TestCase):
    # python -m unittest aggregation_strategies.submodular_mutual_information_strategy_test.TestSubmodularMutualInformation -v
    def test_document_selection(self):
        # Create controlled embeddings
        query_embedding = torch.tensor([1, 0, 0, 0])  # Simplified example embedding
        good_doc_embedding = torch.tensor([0.9, 0.1, 0, 0])  # Similar to query
        bad_doc_embedding = torch.tensor([0, 1, 1, 0])  # Dissimilar to query
        other_doc_embedding = torch.tensor([0.5, 0.5, 0, 0])  # Moderately similar
        document_embeddings = torch.stack(
            [good_doc_embedding, bad_doc_embedding, other_doc_embedding]
        )

        # Apply the function with masks
        good_mask = torch.tensor([True, False, False])
        bad_mask = torch.tensor([False, True, False])

        # Create an instance of SubmodularMutualInformation
        config = config = {
            "architecture": {
                "semantic_search_model": {
                    "checkpoint": "all-mpnet-base-v2",
                    # "device": "cuda:0",
                    "device": "cpu",
                }
            }
        }
        model_ref = WrappedSentenceTransformerModel(config)
        smi_module = SubmodularMutualInformation(config, model_ref)

        good_weighted_avg = smi_module(query_embedding, document_embeddings, good_mask)
        bad_weighted_avg = smi_module(query_embedding, document_embeddings, bad_mask)

        # Assertions
        self.assertTrue(
            torch.norm(good_weighted_avg - query_embedding)
            < torch.norm(bad_weighted_avg - query_embedding)
        )


if __name__ == "__main__":
    unittest.main()
