import unittest
import torch
from train_utils import set_seed
from training_loop_strategies.utils import (
    compute_cutoff_gain,
    compute_dissimilarities,
    compute_search_metrics,
    record_pick,
    select_next_correct,
)


class TestComputeDissimilarities(unittest.TestCase):
    # python -m unittest training_loop_strategies.utils_test.TestComputeDissimilarities.test_no_picked_documents -v
    def test_no_picked_documents(self):
        embeddings = torch.randn(10, 5)
        picked_mask = torch.zeros(10, dtype=torch.bool)
        similarities = torch.randn(10, 5)
        expected_output = torch.zeros(10)

        output = compute_dissimilarities(embeddings, picked_mask, similarities)
        self.assertTrue(torch.equal(output, expected_output))

    # python -m unittest training_loop_strategies.utils_test.TestComputeDissimilarities.test_with_picked_documents -v
    def test_with_picked_documents(self):
        embeddings = torch.randn(10, 5)
        picked_mask = torch.tensor(
            [True, False, True, False, False, False, False, False, False, False]
        )
        similarities = torch.randn(10, 5)

        output = compute_dissimilarities(embeddings, picked_mask, similarities)
        self.assertEqual(output.shape, (10,))


# python -m unittest training_loop_strategies.utils_test.TestSelectNextCorrect -v
class TestSelectNextCorrect(unittest.TestCase):
    # python -m unittest training_loop_strategies.utils_test.TestSelectNextCorrect.test_correct_selection_from_can_be_picked_set -v
    def test_correct_selection_from_can_be_picked_set(self):
        """Test selection when the highest similarity index is in the can_be_picked_set."""
        similarities = torch.tensor([0.1, 0.4, 0.35])
        paraphrase_lut = {0: 2, 1: 0, 2: 1}
        can_be_picked_set = set([1])
        current_picked_mask = torch.tensor([False, True, False])  # Masking index 1

        next_correct = select_next_correct(
            similarities, paraphrase_lut, can_be_picked_set, current_picked_mask
        )

        self.assertEqual(next_correct, 2)  # Should return 2 as 1 is masked

    def test_selection_using_paraphrase_lut(self):
        """Test selection using paraphrase lookup table when the initial selection is not in can_be_picked_set."""
        similarities = torch.tensor([0.6, 0.3, 0.8])
        paraphrase_lut = {0: 2, 2: 0}
        can_be_picked_set = set([0, 1])
        current_picked_mask = torch.tensor([False, False, True])

        next_correct = select_next_correct(
            similarities, paraphrase_lut, can_be_picked_set, current_picked_mask
        )

        # Paraphrase of 2 is 0, which is in can_be_picked_set
        self.assertEqual(next_correct, 0)

    def test_selection_fallback_to_can_be_picked_set(self):
        """Test the fallback mechanism when argmax returns an index not in can_be_picked_set."""
        similarities = torch.tensor([0.0, 0.0, 0.0])
        paraphrase_lut = {}
        can_be_picked_set = set([2])
        current_picked_mask = torch.tensor([True, False, False])

        next_correct = select_next_correct(
            similarities, paraphrase_lut, can_be_picked_set, current_picked_mask
        )

        self.assertEqual(next_correct, 2)

    def test_selection_avoiding_current_picked_mask(self):
        """Test that the selection avoids indices marked in current_picked_mask."""
        similarities = torch.tensor([0.5, 0.2, 0.8])
        paraphrase_lut = {}
        can_be_picked_set = set([0, 1, 2])
        current_picked_mask = torch.tensor([False, True, False])  # Masking index 1

        next_correct = select_next_correct(
            similarities, paraphrase_lut, can_be_picked_set, current_picked_mask
        )

        # Should select index 2 as it has the highest similarity and is not masked
        self.assertEqual(next_correct, 2)


# python -m unittest training_loop_strategies.utils_test.TestSearchMetrics -v
class TestSearchMetrics(unittest.TestCase):
    # python -m unittest training_loop_strategies.utils_test.TestSearchMetrics.test_compute_search_metrics -v
    def test_compute_search_metrics(self):
        config = {"eval": {"k": [1, 2, 3]}}

        # Mock Predictions (Tensor) and convert to ranking indices
        predictions = torch.tensor([0.9, 0.1, 0.4, 0.2, 0.8])
        _, ranking_indices = torch.sort(predictions, descending=True)
        ranking_indices = ranking_indices.tolist()  # [0, 4, 2, 3, 1]

        # Mock Paraphrase Lookup Table
        paraphrase_lut = {0: 1, 1: 0, 2: 4, 4: 2}

        # Mock Set of Relevant Documents
        relevant_doc_ids_without_paraphrase = {0, 2}

        # Expected Results
        expected_metrics = {
            "recall_at_1": 0.5,
            "precision_at_1": 1.0,
            "f1_at_1": 0.6666666666666666,
            "recall_at_2": 1.0,
            "precision_at_2": 1.0,
            "f1_at_2": 1.0,
            "recall_at_3": 1.0,
            "precision_at_3": 0.6666666666666666,
            "f1_at_3": 0.8,
        }

        # Run the function
        actual_metrics = compute_search_metrics(
            config, ranking_indices, paraphrase_lut, relevant_doc_ids_without_paraphrase
        )

        # Assert the results
        for key in expected_metrics:
            self.assertAlmostEqual(
                expected_metrics[key], actual_metrics[key], msg=key, places=2
            )

    # python -m unittest training_loop_strategies.utils_test.TestSearchMetrics.test_compute_search_metrics_with_k_out_of_range -v
    def test_compute_search_metrics_with_k_out_of_range(self):
        # Mock Config with k larger than the number of predictions
        config = {"eval": {"k": [1, 5]}}

        # Mock Predictions (Tensor) with only 3 items and convert to ranking indices
        predictions = torch.tensor([0.9, 0.8, 0.7])
        _, ranking_indices = torch.sort(predictions, descending=True)
        ranking_indices = ranking_indices.tolist()  # [0, 1, 2]

        # Mock Paraphrase Lookup Table
        paraphrase_lut = {0: 1, 1: 0}

        # Mock Set of Relevant Documents
        relevant_doc_ids_without_paraphrase = {0, 2}

        expected_metrics = {
            "recall_at_1": 0.5,
            "precision_at_1": 1.0,
            "f1_at_1": 0.6666666666666666,
            "recall_at_5": 1.0,
            "precision_at_5": 0.6666666666666666,
            "f1_at_5": 0.8,
        }

        # Run the function
        actual_metrics = compute_search_metrics(
            config, ranking_indices, paraphrase_lut, relevant_doc_ids_without_paraphrase
        )

        # Assert the results
        for key in expected_metrics:
            self.assertAlmostEqual(
                expected_metrics[key], actual_metrics[key], msg=key, places=2
            )


# python -m unittest training_loop_strategies.utils_test.TestRecordPick -v
class TestRecordPick(unittest.TestCase):
    def test_record_pick(self):
        # Sample inputs for testing
        next_correct = 2
        can_be_picked_set = {1, 2}
        paraphrase_lut = {1: 4, 2: 5, 4: 1, 5: 2}
        labels_mask_list = [torch.tensor([False, True, True, False, True, True])]
        picked_mask_list = [torch.tensor([False, False, False, False, False, False])]
        teacher_forcing = []

        # Expected outputs
        expected_can_be_picked_set = {1}
        expected_labels_mask_list = [
            torch.tensor([False, True, True, False, True, True]),
            torch.tensor([False, True, False, False, True, False]),
        ]
        expected_picked_mask_list = [
            torch.tensor([False, False, False, False, False, False]),
            torch.tensor([False, False, True, False, False, False]),
        ]
        expected_teacher_forcing = [2]

        # Call the function to test
        record_pick(
            next_correct,
            can_be_picked_set,
            paraphrase_lut,
            labels_mask_list,
            picked_mask_list,
            teacher_forcing,
        )

        self.assertEqual(can_be_picked_set, expected_can_be_picked_set)
        self.assertEqual(teacher_forcing, expected_teacher_forcing)
        self.assertEqual(len(labels_mask_list), len(expected_labels_mask_list))
        self.assertEqual(len(picked_mask_list), len(expected_picked_mask_list))
        for i in range(len(labels_mask_list)):
            self.assertEqual(
                labels_mask_list[i].tolist(),
                expected_labels_mask_list[i].tolist(),
            )
            self.assertEqual(
                picked_mask_list[i].tolist(), expected_picked_mask_list[i].tolist()
            )

        ### Iteration 2 ###

        # Expected outputs
        next_correct = 4
        expected_can_be_picked_set = set()
        expected_labels_mask_list = [
            torch.tensor([False, True, True, False, True, True]),
            torch.tensor([False, True, False, False, True, False]),
            torch.tensor([False, False, False, False, False, False]),
        ]
        expected_picked_mask_list = [
            torch.tensor([False, False, False, False, False, False]),
            torch.tensor([False, False, True, False, False, False]),
            torch.tensor([False, False, True, False, True, False]),
        ]
        expected_teacher_forcing = [2, 4]

        # Call the function to test
        record_pick(
            next_correct,
            can_be_picked_set,
            paraphrase_lut,
            labels_mask_list,
            picked_mask_list,
            teacher_forcing,
        )

        self.assertEqual(can_be_picked_set, expected_can_be_picked_set)
        self.assertEqual(teacher_forcing, expected_teacher_forcing)
        self.assertEqual(len(labels_mask_list), len(expected_labels_mask_list))
        self.assertEqual(len(picked_mask_list), len(expected_picked_mask_list))
        for i in range(len(labels_mask_list)):
            self.assertEqual(
                labels_mask_list[i].tolist(),
                expected_labels_mask_list[i].tolist(),
            )
            self.assertEqual(
                picked_mask_list[i].tolist(), expected_picked_mask_list[i].tolist()
            )


# python -m unittest training_loop_strategies.utils_test.TestComputeCutoffGain -v
class TestComputeCutoffGain(unittest.TestCase):
    # python -m unittest training_loop_strategies.utils_test.TestComputeCutoffGain.test_no_picked_documents -v
    def test_no_picked_documents(self):
        predictions = torch.tensor([0.1, 0.2, 0.3, 0.4])
        global_correct_mask = torch.tensor([True, True, False, False])
        current_picked_mask = torch.tensor([False, False, False, False])
        paraphrase_lut = {0: 1}

        result = compute_cutoff_gain(
            predictions,
            global_correct_mask,  # dont clone for testing
            current_picked_mask,
            paraphrase_lut,
        )
        self.assertAlmostEqual(result, -0.3, places=4)
        self.assertTrue(
            torch.equal(global_correct_mask, torch.tensor([True, True, False, False]))
        )

    # python -m unittest training_loop_strategies.utils_test.TestComputeCutoffGain.test_picked_documents -v
    def test_picked_documents(self):
        predictions = torch.tensor([0.1, 0.4, 0.3, 0.2])
        global_correct_mask = torch.tensor([False, True, True, False])
        current_picked_mask = torch.tensor([False, True, False, False])
        paraphrase_lut = {1: 2}

        result = compute_cutoff_gain(
            predictions,
            global_correct_mask,  # dont clone for testing
            current_picked_mask,
            paraphrase_lut,
        )
        self.assertAlmostEqual(result, 0.1, places=4)
        self.assertTrue(
            torch.equal(global_correct_mask, torch.tensor([False, True, False, False]))
        )

    # python -m unittest training_loop_strategies.utils_test.TestComputeCutoffGain.test_empty_most_similar_incorrect -v
    def test_empty_most_similar_incorrect(self):
        # Set up a scenario where all predictions are marked as correct
        predictions = torch.tensor([0.2, 0.5, 0.3, 0.4])
        global_correct_mask = torch.tensor([True, True, True, True])
        current_picked_mask = torch.tensor([False, False, False, False])
        paraphrase_lut = {0: 1, 2: 3}

        gain = compute_cutoff_gain(
            predictions, global_correct_mask, current_picked_mask, paraphrase_lut
        )
        self.assertAlmostEqual(gain, 0.0, places=4)

        predictions = torch.tensor([0.2, 0.5, 0.3, 0.4])
        global_correct_mask = torch.tensor([False, False, False, False])
        current_picked_mask = torch.tensor([False, False, False, False])
        paraphrase_lut = {}

        gain = compute_cutoff_gain(
            predictions, global_correct_mask, current_picked_mask, paraphrase_lut
        )
        self.assertAlmostEqual(gain, 0.0, places=4)


if __name__ == "__main__":
    unittest.main()
