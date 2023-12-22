from aggregation_strategies.submodular_mutual_information_strategy import (
    SubmodularMutualInformation,
)
from models.wrapped_sentence_transformer import WrappedSentenceTransformerModel
import torch
import unittest


# python -m unittest aggregation_strategies.submodular_mutual_information_strategy_test.TestSubmodularMutualInformation -v
class TestSubmodularMutualInformation(unittest.TestCase):
    # python -m unittest aggregation_strategies.submodular_mutual_information_strategy_test.TestSubmodularMutualInformation.test_document_selection -v
    def test_document_selection(self):
        # Create controlled embeddings
        query_embedding = torch.tensor([1.0, 0.0, 0.0, 0.0])
        good_doc_embedding = torch.tensor([0.9, 0.1, 0, 0])  # Similar to query
        bad_doc_embedding = torch.tensor([0.0, 1.0, 1.0, 0.0])  # Dissimilar to query
        other_doc_embedding = torch.tensor([0.5, 0.5, 0, 0])  # Moderately similar
        document_embeddings = torch.stack(
            [good_doc_embedding, bad_doc_embedding, other_doc_embedding]
        )

        # Apply the function with masks
        good_mask = torch.tensor([True, False, False])
        bad_mask = torch.tensor([False, True, False])

        # Create an instance of SubmodularMutualInformation
        config = {
            "architecture": {
                "semantic_search_model": {
                    "checkpoint": "all-mpnet-base-v2",
                    # "device": "cuda:0",
                    "device": "cpu",
                },
                "aggregation_strategy": {
                    "merge_strategy": {"name": "weighted_average_merger"}
                },
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

    # python -m unittest aggregation_strategies.submodular_mutual_information_strategy_test.TestSubmodularMutualInformation.test_disable_q -v
    def test_disable_q(self):
        # Create controlled embeddings
        query_embedding = torch.tensor([1.0, 0.0, 0.0, 0.0])
        good_doc_embedding = torch.tensor([0.9, 0.1, 0, 0])  # Similar to query
        bad_doc_embedding = torch.tensor([0.0, 1.0, 1.0, 0.0])  # Dissimilar to query
        other_doc_embedding = torch.tensor([0.5, 0.5, 0, 0])  # Moderately similar
        document_embeddings = torch.stack(
            [good_doc_embedding, bad_doc_embedding, other_doc_embedding]
        )

        # Apply the function with masks
        mask = torch.tensor([True, False, False])

        # Create an instance of SubmodularMutualInformation
        config = {
            "architecture": {
                "semantic_search_model": {
                    "checkpoint": "all-mpnet-base-v2",
                    # "device": "cuda:0",
                    "device": "cpu",
                },
                "aggregation_strategy": {
                    "quality_gain": {"disabled": True},
                    "merge_strategy": {"name": "weighted_average_merger"},
                },
            }
        }
        model_ref = WrappedSentenceTransformerModel(config)
        smi_module = SubmodularMutualInformation(config, model_ref)

        # Should not crash
        smi_module(query_embedding, document_embeddings, mask)

        # TODO: Test for disabling

    # python -m unittest aggregation_strategies.submodular_mutual_information_strategy_test.TestSubmodularMutualInformation.test_disable_d -v
    def test_disable_d(self):
        # Create controlled embeddings
        query_embedding = torch.tensor([1.0, 0.0, 0.0, 0.0])
        good_doc_embedding = torch.tensor([0.9, 0.1, 0, 0])  # Similar to query
        bad_doc_embedding = torch.tensor([0.0, 1.0, 1.0, 0.0])  # Dissimilar to query
        other_doc_embedding = torch.tensor([0.5, 0.5, 0, 0])  # Moderately similar
        document_embeddings = torch.stack(
            [good_doc_embedding, bad_doc_embedding, other_doc_embedding]
        )

        # Apply the function with masks
        mask = torch.tensor([True, False, False])

        # Create an instance of SubmodularMutualInformation
        config = {
            "architecture": {
                "semantic_search_model": {
                    "checkpoint": "all-mpnet-base-v2",
                    # "device": "cuda:0",
                    "device": "cpu",
                },
                "aggregation_strategy": {
                    "diversity_gain": {"disabled": True},
                    "merge_strategy": {"name": "weighted_average_merger"},
                },
            }
        }
        model_ref = WrappedSentenceTransformerModel(config)
        smi_module = SubmodularMutualInformation(config, model_ref)

        # Should not crash
        smi_module(query_embedding, document_embeddings, mask)

        # TODO: Test for disabling

    # python -m unittest aggregation_strategies.submodular_mutual_information_strategy_test.TestSubmodularMutualInformation.test_disable_q_and_d -v
    def test_disable_q_and_d(self):
        # Create controlled embeddings
        query_embedding = torch.tensor([1.0, 0.0, 0.0, 0.0])
        good_doc_embedding = torch.tensor([0.9, 0.1, 0, 0])  # Similar to query
        bad_doc_embedding = torch.tensor([0.0, 1.0, 1.0, 0.0])  # Dissimilar to query
        other_doc_embedding = torch.tensor([0.5, 0.5, 0, 0])  # Moderately similar
        document_embeddings = torch.stack(
            [good_doc_embedding, bad_doc_embedding, other_doc_embedding]
        )

        # Apply the function with masks
        mask = torch.tensor([True, False, False])

        # Create an instance of SubmodularMutualInformation
        config = {
            "architecture": {
                "semantic_search_model": {
                    "checkpoint": "all-mpnet-base-v2",
                    # "device": "cuda:0",
                    "device": "cpu",
                },
                "aggregation_strategy": {
                    "quality_gain": {"disabled": True},
                    "diversity_gain": {"disabled": True},
                    "merge_strategy": {"name": "weighted_average_merger"},
                },
            }
        }
        model_ref = WrappedSentenceTransformerModel(config)
        with self.assertRaises(ValueError):
            SubmodularMutualInformation(config, model_ref)

    # python -m unittest aggregation_strategies.submodular_mutual_information_strategy_test.TestSubmodularMutualInformation.test_document_selection_error_case -v
    def test_document_selection_error_case(self):
        # Create controlled embeddings
        query_embedding = torch.rand(768)  # Random embedding of size 768
        document_embeddings = torch.rand(42, 768)  # 42 random document embeddings

        # Create a mask that selects some documents, leading to the error
        # Selects 2 documents, leading to size mismatch
        mask = torch.tensor([True] * 2 + [False] * 40)

        # Create an instance of SubmodularMutualInformation
        config = {
            "architecture": {
                "semantic_search_model": {
                    "checkpoint": "all-mpnet-base-v2",
                    # "device": "cuda:0",
                    "device": "cpu",
                },
                "aggregation_strategy": {
                    "merge_strategy": {"name": "weighted_average_merger"}
                },
            }
        }
        model_ref = WrappedSentenceTransformerModel(config)
        smi_module = SubmodularMutualInformation(config, model_ref)

        # Should not crash
        smi_module(query_embedding, document_embeddings, mask)


# python -m unittest aggregation_strategies.submodular_mutual_information_strategy_test.TestQualityGainSequential -v
class TestQualityGainSequential(unittest.TestCase):
    def setUp(self):
        config = {
            "architecture": {
                "semantic_search_model": {
                    "checkpoint": "all-mpnet-base-v2",
                    # "device": "cuda:0",
                    "device": "cpu",
                }
            }
        }

        class MockMerger:
            def __init__(self):
                self.merge_model = lambda x, y: x * y

        model_ref = MockMerger()
        self.instance = SubmodularMutualInformation(config, model_ref)

    # python -m unittest aggregation_strategies.submodular_mutual_information_strategy_test.TestQualityGainSequential.test_basic_functionality -v
    def test_basic_functionality(self):
        # Mock input data
        question_embedding = torch.randn(1, 10)
        filtered_document_embeddings = torch.randn(5, 10)

        # Call your function
        result = self.instance._quality_gain_sequential(
            question_embedding, filtered_document_embeddings
        )

        # Check the shape of the result
        self.assertEqual(result.shape, (5, 5))

    # python -m unittest aggregation_strategies.submodular_mutual_information_strategy_test.TestQualityGainSequential.test_small_interpretable_input -v
    def test_small_interpretable_input(self):
        # Small, fixed input data
        question_embedding = torch.tensor([[1.0, 2.0]])
        filtered_document_embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

        # Expected output (gain is symetric)
        expected_output = torch.tensor([[0.0, 0.0], [0.0, 0.0]])

        # Call your function
        result = self.instance._quality_gain_sequential(
            question_embedding, filtered_document_embeddings
        )

        # Check if the output matches the expected output
        self.assertTrue(torch.allclose(result, expected_output, atol=1e-6))


# python -m unittest aggregation_strategies.submodular_mutual_information_strategy_test.TestQualityGain -v
class TestQualityGain(unittest.TestCase):
    def setUp(self):
        config = {
            "architecture": {
                "semantic_search_model": {
                    "checkpoint": "all-mpnet-base-v2",
                    # "device": "cuda:0",
                    "device": "cpu",
                }
            }
        }

        class MockMerger:
            def __init__(self):
                self.merge_model = lambda x, y: x * y

        model_ref = MockMerger()
        self.instance = SubmodularMutualInformation(config, model_ref)

    # python -m unittest aggregation_strategies.submodular_mutual_information_strategy_test.TestQualityGain.test_basic_functionality -v
    def test_basic_functionality(self):
        # Mock input data
        question_embedding = torch.randn(1, 10)
        filtered_document_embeddings = torch.randn(5, 10)

        # Call your function
        result = self.instance._quality_gain(
            question_embedding, filtered_document_embeddings
        )

        expected_output = self.instance._quality_gain_sequential(
            question_embedding, filtered_document_embeddings
        )

        # Check the shape of the result
        self.assertEqual(result.shape, (5, 5))

        self.assertTrue(torch.allclose(result, expected_output, atol=1e-6))

    # python -m unittest aggregation_strategies.submodular_mutual_information_strategy_test.TestQualityGain.test_small_interpretable_input -v
    def test_small_interpretable_input(self):
        # Small, fixed input data
        question_embedding = torch.tensor([[1.0, 2.0]])
        filtered_document_embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

        # Expected output (gain is symetric)
        expected_output = torch.tensor([[0.0, 0.0], [0.0, 0.0]])

        # Call your function
        result = self.instance._quality_gain(
            question_embedding, filtered_document_embeddings
        )

        # Check if the output matches the expected output
        self.assertTrue(torch.allclose(result, expected_output, atol=1e-6))


# python -m unittest aggregation_strategies.submodular_mutual_information_strategy_test.TestDiversityGainSequential -v
class TestDiversityGainSequential(unittest.TestCase):
    def setUp(self):
        config = {
            "architecture": {
                "semantic_search_model": {
                    "checkpoint": "all-mpnet-base-v2",
                    # "device": "cuda:0",
                    "device": "cpu",
                }
            }
        }

        class MockMerger:
            def __init__(self):
                self.merge_model = lambda x, y: x * y

        model_ref = MockMerger()
        self.instance = SubmodularMutualInformation(config, model_ref)

    # python -m unittest aggregation_strategies.submodular_mutual_information_strategy_test.TestDiversityGainSequential.test_basic_functionality -v
    def test_basic_functionality(self):
        # Mock input data
        question_embedding = torch.randn(1, 10)
        filtered_document_embeddings = torch.randn(5, 10)

        # Call your function
        result = self.instance._diversity_gain_sequential(
            question_embedding, filtered_document_embeddings
        )

        # Check the shape of the result
        self.assertEqual(result.shape, (5, 5))

    # python -m unittest aggregation_strategies.submodular_mutual_information_strategy_test.TestDiversityGainSequential.test_small_interpretable_input -v
    def test_small_interpretable_input(self):
        # Small, fixed input data
        question_embedding = torch.tensor([[1.0, 2.0]])
        filtered_document_embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

        # Expected output (gain is symetric)
        expected_output = torch.tensor([[0.0, 0.0], [0.0, 0.0]])

        # Call your function
        result = self.instance._diversity_gain_sequential(
            question_embedding, filtered_document_embeddings
        )

        # Check if the output matches the expected output
        self.assertTrue(torch.allclose(result, expected_output, atol=1e-6))


# python -m unittest aggregation_strategies.submodular_mutual_information_strategy_test.TestDiversityGain -v
class TestDiversityGain(unittest.TestCase):
    def setUp(self):
        config = {
            "architecture": {
                "semantic_search_model": {
                    "checkpoint": "all-mpnet-base-v2",
                    # "device": "cuda:0",
                    "device": "cpu",
                }
            }
        }

        class MockMerger:
            def __init__(self):
                self.merge_model = lambda x, y: x * y

        model_ref = MockMerger()
        self.instance = SubmodularMutualInformation(config, model_ref)

    # python -m unittest aggregation_strategies.submodular_mutual_information_strategy_test.TestDiversityGain.test_basic_functionality -v
    def test_basic_functionality(self):
        # Mock input data
        question_embedding = torch.randn(1, 10)
        filtered_document_embeddings = torch.randn(5, 10)

        # Call your function
        result = self.instance._diversity_gain(
            question_embedding, filtered_document_embeddings
        )

        expected_output = self.instance._diversity_gain_sequential(
            question_embedding, filtered_document_embeddings
        )

        # Check the shape of the result
        self.assertEqual(result.shape, (5, 5))

        self.assertTrue(torch.allclose(result, expected_output, atol=1e-6))

    # python -m unittest aggregation_strategies.submodular_mutual_information_strategy_test.TestDiversityGain.test_small_interpretable_input -v
    def test_small_interpretable_input(self):
        # Small, fixed input data
        question_embedding = torch.tensor([[1.0, 2.0]])
        filtered_document_embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

        # Expected output (gain is symetric)
        expected_output = torch.tensor([[0.0, 0.0], [0.0, 0.0]])

        # Call your function
        result = self.instance._diversity_gain(
            question_embedding, filtered_document_embeddings
        )

        # Check if the output matches the expected output
        self.assertTrue(torch.allclose(result, expected_output, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
