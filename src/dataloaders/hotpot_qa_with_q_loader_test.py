import unittest
from dataloaders.hotpot_qa_with_q_loader import add_question_to_row
from datasets import load_dataset
from models.t5_model import T5ModelForQuestionGeneration
from dataloaders.hotpot_qa_with_q_loader import get_loader
from train_utils import set_seed
import numpy as np


# python -m unittest dataloaders.hotpot_qa_with_q_loader_test.TestAddQuestionToRow -v
@unittest.skip("needs gpu")
class TestAddQuestionToRow(unittest.TestCase):
    # python -m unittest dataloaders.hotpot_qa_with_q_loader_test.TestAddQuestionToRow.test_with_t5 -v
    def test_with_t5(self):
        dataset = load_dataset("hotpot_qa", "distractor")
        row = dataset["train"][0]

        config = {
            "architecture": {
                "question_generator_model": {
                    "name": "t5",
                    "size": "base",
                    "device": "cuda:0",
                }
            }
        }
        model = T5ModelForQuestionGeneration(config)

        modified_row = add_question_to_row(model, row)

        assert modified_row["questions"] is not None


# python -m unittest dataloaders.hotpot_qa_with_q_loader_test.TestHotpotQaWithQaLoader -v
class TestHotpotQaWithQaLoader(unittest.TestCase):
    # python -m unittest dataloaders.hotpot_qa_with_q_loader_test.TestHotpotQaWithQaLoader.test_happy_path -v
    def test_happy_path(self):
        # Set seed for deterministic testing
        set_seed(42)  # TODO: shuffle with seed

        batch_size = 1

        train_loader, val_loader = get_loader(batch_size)

        # Train loader
        batch = next(iter(train_loader))
        expected = (
            "Which magazine was started first Arthur's Magazine or First for Women?"
        )
        actual = batch["questions"][0]
        assert actual == expected, actual

        expected = [
            "Arthur's Magazine (1844–1846) was an American literary periodical published in Philadelphia in the 19th century.",
            "First for Women is a woman's magazine published by Bauer Media Group in the USA.",
        ]
        actual = list(
            np.array(batch["flat_sentences"][0])[batch["relevant_sentence_indexes"][0]]
        )
        assert expected == actual, actual

        expected = [
            "What was the significance of Arthur's Magazine in the American literary landscape of the 19th century?",
            "What is the name of the magazine published by Bauer Media Group in the USA that specifically targets women?",
        ]
        actual = list(
            np.array(batch["flat_questions"][0])[batch["relevant_sentence_indexes"][0]]
        )
        assert expected == actual, actual

        # Validation loader
        batch = next(iter(val_loader))
        expected = "Were Scott Derrickson and Ed Wood of the same nationality?"
        actual = batch["questions"][0]
        assert actual == expected, actual

        expected = [
            "Scott Derrickson (born July 16, 1966) is an American director, screenwriter and producer.",
            "Edward Davis Wood Jr. (October 10, 1924 – December 10, 1978) was an American filmmaker, actor, writer, producer, and director.",
        ]
        actual = list(
            np.array(batch["flat_sentences"][0])[batch["relevant_sentence_indexes"][0]]
        )
        assert expected == actual, actual

        expected = [
            "What is Scott Derrickson known for in the entertainment industry?",
            "What were some of the notable contributions and achievements of Edward Davis Wood Jr. in the field of filmmaking?",
        ]
        actual = list(
            np.array(batch["flat_questions"][0])[batch["relevant_sentence_indexes"][0]]
        )
        assert expected == actual, actual


if __name__ == "__main__":
    unittest.main()
