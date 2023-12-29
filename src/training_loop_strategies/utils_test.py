import unittest
import torch
from train_utils import set_seed
from training_loop_strategies.utils import (
    compute_cutoff_gain,
    compute_dissimilarities,
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
    def test_correct_selection_from_can_be_picked_set(self):
        similarities = torch.tensor([0.1, 0.4, 0.35])
        paraphrase_lut = {0: 2, 1: 0, 2: 1}
        recall_at_1 = 0
        can_be_picked_set = set([1])
        selected_index = 1

        next_correct, updated_recall = select_next_correct(
            similarities, paraphrase_lut, recall_at_1, can_be_picked_set, selected_index
        )

        self.assertEqual(next_correct, selected_index)
        self.assertEqual(updated_recall, recall_at_1 + 1)

    def test_update_recall_at_1(self):
        similarities = torch.tensor([0.1, 0.2, 0.3])
        paraphrase_lut = {0: 1, 1: 2, 2: 0}
        recall_at_1 = 5
        can_be_picked_set = set([1, 2])
        selected_index = 0  # Paraphrase of 0 is 1, which is in can_be_picked_set

        _, updated_recall = select_next_correct(
            similarities, paraphrase_lut, recall_at_1, can_be_picked_set, selected_index
        )

        self.assertEqual(updated_recall, recall_at_1 + 1)

    def test_selection_using_similarities(self):
        similarities = torch.tensor([0.1, 0.7, 0.6])
        paraphrase_lut = {}
        recall_at_1 = 0
        can_be_picked_set = set([1, 2])
        selected_index = 0  # Not in can_be_picked_set

        next_correct, _ = select_next_correct(
            similarities, paraphrase_lut, recall_at_1, can_be_picked_set, selected_index
        )

        self.assertEqual(next_correct, 1)  # Highest similarity in can_be_picked_set

    def test_edge_case_for_argmax_zero(self):
        similarities = torch.tensor([0.0, 0.0, 0.0])
        paraphrase_lut = {}
        recall_at_1 = 0
        can_be_picked_set = set([2])
        selected_index = 0

        next_correct, _ = select_next_correct(
            similarities, paraphrase_lut, recall_at_1, can_be_picked_set, selected_index
        )

        self.assertEqual(next_correct, 2)  # Falls back to first in can_be_picked_set

    def test_function_with_various_inputs(self):
        similarities = torch.tensor([0.5, 0.2, 0.8])
        paraphrase_lut = {0: 2, 2: 0}
        recall_at_1 = 3
        can_be_picked_set = set([0, 1])
        selected_index = 2  # Paraphrase of 2 is 0, which is in can_be_picked_set

        next_correct, updated_recall = select_next_correct(
            similarities, paraphrase_lut, recall_at_1, can_be_picked_set, selected_index
        )

        self.assertEqual(next_correct, 2)
        self.assertEqual(updated_recall, recall_at_1 + 1)


# python -m unittest training_loop_strategies.utils_test.TestRecordPick -v
class TestRecordPick(unittest.TestCase):
    # python -m unittest training_loop_strategies.utils_test.TestRecordPick.test_removal_from_can_be_picked_set -v
    def test_removal_from_can_be_picked_set(self):
        next_correct = 2
        can_be_picked_set = set([1, 2, 3])
        paraphrase_lut = {1: 4, 2: 5, 3: 6}
        # Ensure the tensors are large enough for the paraphrase indices
        current_all_selection_vector = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        current_picked_mask = torch.tensor(
            [False, False, False, False, False, False, False]
        )
        all_selection_vector_list = []
        picked_mask_list = []
        teacher_forcing = []

        record_pick(
            next_correct,
            can_be_picked_set,
            paraphrase_lut,
            current_all_selection_vector,
            all_selection_vector_list,
            current_picked_mask,
            picked_mask_list,
            teacher_forcing,
        )

        self.assertNotIn(next_correct, can_be_picked_set)

    # python -m unittest training_loop_strategies.utils_test.TestRecordPick.test_update_all_selection_vector_list -v
    def test_update_all_selection_vector_list(self):
        next_correct = 2
        can_be_picked_set = set([1, 2, 3])
        paraphrase_lut = {1: 4, 2: 5, 3: 6}
        # Ensure the tensors are large enough for the paraphrase indices
        current_all_selection_vector = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        current_picked_mask = torch.tensor(
            [False, False, False, False, False, False, False]
        )
        all_selection_vector_list = []
        picked_mask_list = []
        teacher_forcing = []

        record_pick(
            next_correct,
            can_be_picked_set,
            paraphrase_lut,
            current_all_selection_vector,
            all_selection_vector_list,
            current_picked_mask,
            picked_mask_list,
            teacher_forcing,
        )

        self.assertEqual(len(all_selection_vector_list), 1)
        self.assertTrue(
            torch.equal(
                all_selection_vector_list[0],
                torch.tensor([0.1, 0.2, 0, 0.4, 0.5, 0, 0.7]),
            )
        )

    # python -m unittest training_loop_strategies.utils_test.TestRecordPick.test_update_picked_mask_list -v
    def test_update_picked_mask_list(self):
        next_correct = 2
        can_be_picked_set = set([1, 2, 3])
        paraphrase_lut = {1: 4, 2: 5, 3: 6}
        # Ensure the tensors are large enough for the paraphrase indices
        current_all_selection_vector = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        current_picked_mask = torch.tensor(
            [False, False, False, False, False, False, False]
        )
        all_selection_vector_list = []
        picked_mask_list = []
        teacher_forcing = []

        record_pick(
            next_correct,
            can_be_picked_set,
            paraphrase_lut,
            current_all_selection_vector,
            all_selection_vector_list,
            current_picked_mask,
            picked_mask_list,
            teacher_forcing,
        )

        self.assertEqual(len(picked_mask_list), 1)
        self.assertTrue(
            torch.equal(
                picked_mask_list[0],
                torch.tensor([False, False, True, False, False, False, False]),
            ),
        )

    # python -m unittest training_loop_strategies.utils_test.TestRecordPick.test_update_teacher_forcing_list -v
    def test_update_teacher_forcing_list(self):
        next_correct = 2
        can_be_picked_set = set([1, 2, 3])
        paraphrase_lut = {1: 4, 2: 5, 3: 6}
        # Ensure the tensors are large enough for the paraphrase indices
        current_all_selection_vector = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        current_picked_mask = torch.tensor(
            [False, False, False, False, False, False, False]
        )
        all_selection_vector_list = []
        picked_mask_list = []
        teacher_forcing = []

        record_pick(
            next_correct,
            can_be_picked_set,
            paraphrase_lut,
            current_all_selection_vector,
            all_selection_vector_list,
            current_picked_mask,
            picked_mask_list,
            teacher_forcing,
        )

        self.assertEqual(teacher_forcing, [next_correct])


# python -m unittest training_loop_strategies.utils_test.TestComputeCutoffGain -v
class TestComputeCutoffGain(unittest.TestCase):
    # python -m unittest training_loop_strategies.utils_test.TestComputeCutoffGain.test_no_picked_documents -v
    def test_no_picked_documents(self):
        similarities = torch.tensor([0.1, 0.2, 0.3, 0.4])
        global_correct_mask = torch.tensor([True, True, False, True])
        current_picked_mask = torch.tensor([False, False, False, False])
        paraphrase_lut = {}

        result = compute_cutoff_gain(
            similarities,
            global_correct_mask,
            current_picked_mask,
            paraphrase_lut,
        )
        self.assertAlmostEqual(result, -0.2, places=4)
        self.assertTrue(
            torch.equal(global_correct_mask, torch.tensor([True, True, False, True]))
        )

    def test_picked_documents_no_paraphrases(self):
        similarities = torch.tensor([0.1, 0.4, 0.3, 0.2])
        global_correct_mask = torch.tensor([True, True, True, False])
        current_picked_mask = torch.tensor([False, True, False, False])
        paraphrase_lut = {}

        result = compute_cutoff_gain(
            similarities,
            global_correct_mask,
            current_picked_mask,
            paraphrase_lut,
        )
        self.assertAlmostEqual(result, -0.1, places=4)
        self.assertTrue(
            torch.equal(global_correct_mask, torch.tensor([True, True, True, False]))
        )

    def test_picked_paraphrase_documents(self):
        similarities = torch.tensor([0.5, 0.2, 0.3, 0.1])
        global_correct_mask = torch.tensor([True, False, True, True])
        current_picked_mask = torch.tensor([True, False, False, False])
        paraphrase_lut = {0: 2}  # Document 0 is a paraphrase of Document 2

        result = compute_cutoff_gain(
            similarities,
            global_correct_mask,
            current_picked_mask,
            paraphrase_lut,
        )
        self.assertAlmostEqual(result, -0.2, places=4)
        # global_correct_mask should be updated since document 0 (a paraphrase of 2) is picked
        self.assertTrue(
            torch.equal(global_correct_mask, torch.tensor([True, False, False, True]))
        )


# python -m unittest training_loop_strategies.utils_test.TestIntegration -v
class TestIntegration(unittest.TestCase):
    # python -m unittest training_loop_strategies.utils_test.TestIntegration.test_full_flow -v
    def test_full_flow(self):
        # Setup test data
        set_seed(42)
        document_embeddings = torch.randn(10, 5)  # Simulated document embeddings
        query_embedding = torch.randn(1, 5)  # Simulated query embedding
        flat_questions = list(range(10))  # Simulated flat question indices
        # Indices without paraphrases
        no_paraphrase_relevant_question_indexes = set([1, 3, 5, 7])
        num_correct = len(no_paraphrase_relevant_question_indexes)
        paraphrase_lut = {0: 2, 2: 0, 4: 6, 6: 4}  # Paraphrase look-up table

        # Compute initial similarities
        similarities = torch.matmul(document_embeddings, query_embedding.T).squeeze()
        similarities = torch.clamp(similarities, min=0, max=1)

        # Initialize vectors and masks
        selection_vector = torch.ones(len(flat_questions))  # Initial selection vector
        picked_mask = torch.zeros(len(flat_questions), dtype=torch.bool)
        selection_vector_list = [selection_vector.clone()]
        picked_mask_list = [picked_mask]
        teacher_forcing = []
        recall_at_1 = 0
        total_loss = torch.zeros([])

        # Processing loop
        for _ in range(len(no_paraphrase_relevant_question_indexes)):
            current_all_selection_vector = selection_vector_list[-1]
            current_picked_mask = picked_mask_list[-1]

            # Function calls
            dissimilarities = compute_dissimilarities(
                document_embeddings, current_picked_mask, similarities
            )
            # Assuming a loss function is defined
            loss = torch.nn.functional.mse_loss(
                similarities, current_all_selection_vector.float()
            )
            total_loss += loss

            cloned_predictions = similarities * (1 - dissimilarities)
            cloned_predictions[current_picked_mask] = 0
            selected_index = torch.argmax(cloned_predictions).item()

            next_correct, recall_at_1 = select_next_correct(
                similarities,
                paraphrase_lut,
                recall_at_1,
                no_paraphrase_relevant_question_indexes,
                selected_index,
            )

            record_pick(
                next_correct,
                no_paraphrase_relevant_question_indexes,
                paraphrase_lut,
                current_all_selection_vector,
                selection_vector_list,
                current_picked_mask,
                picked_mask_list,
                teacher_forcing,
            )

        # Assertions to validate the flow
        self.assertEqual(len(teacher_forcing), num_correct)
        self.assertEqual(len(no_paraphrase_relevant_question_indexes), 0)
        self.assertEqual(recall_at_1, 0)


if __name__ == "__main__":
    unittest.main()
