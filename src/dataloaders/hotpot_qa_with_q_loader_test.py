import unittest
from dataloaders.hotpot_qa_with_q_loader import get_train_loader, get_validation_loader
from tqdm import tqdm
from train_utils import set_seed
import numpy as np


# python -m unittest dataloaders.hotpot_qa_with_q_loader_test.TestHotpotQaWithQaLoader -v
class TestHotpotQaWithQaLoader(unittest.TestCase):
    # python -m unittest dataloaders.hotpot_qa_with_q_loader_test.TestHotpotQaWithQaLoader.test_happy_path -v
    def test_happy_path(self):
        # Set seed for deterministic testing
        set_seed(42)

        batch_size = 1

        train_loader = get_train_loader(batch_size=batch_size)
        val_loader = get_validation_loader(batch_size=batch_size)

        # Train loader
        batch = next(iter(train_loader))
        expected = "Who was born first, William March or Richard Brautigan?"
        actual = batch["questions"][0]
        assert actual == expected, actual

        expected = [
            "William March (September 18, 1893 – May 15, 1954) was an American writer of psychological fiction and a highly decorated US Marine.",
            "Richard Gary Brautigan (January 30, 1935 – ca.",
        ]
        actual = list(
            np.array(batch["flat_sentences"][0])[batch["relevant_sentence_indexes"][0]]
        )
        assert expected == actual, actual

        expected = [
            "What were some notable achievements of William March as both a writer and a US Marine?",
            "What is the significance of Richard Brautigan's work in the literary world?",
            "What were some significant accomplishments of William March in his roles as a writer and a member of the US Marine Corps?",
            "Why is Richard Brautigan's work important in the field of literature?",
        ]
        actual = list(
            np.array(batch["flat_questions"][0])[batch["relevant_question_indexes"][0]]
        )
        assert expected == actual, actual
        actual = list(
            np.array(batch["flat_questions"][0])[batch["selection_vector"][0]]
        )
        assert expected == actual, actual

        paraphrase_lut = batch["paraphrase_lut"][0]
        flat_questions = batch["flat_questions"][0]
        expected = [
            [
                "What were some notable achievements of William March as both a writer and a US Marine?",
                "What were some significant accomplishments of William March in his roles as a writer and a member of the US Marine Corps?",
            ],
            [
                "What is the significance of Richard Brautigan's work in the literary world?",
                "Why is Richard Brautigan's work important in the field of literature?",
            ],
        ]
        for (expected_left, expected_right), (key_left, key_right) in zip(
            expected, list(paraphrase_lut.items())[::2]
        ):
            assert expected_left == flat_questions[key_left], flat_questions[key_left]
            assert expected_right == flat_questions[key_right], flat_questions[
                key_right
            ]

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
            "What is Scott Derrickson's claim to fame in the entertainment industry?",
            "What were some of the significant accomplishments and contributions made by Edward Davis Wood Jr. in the realm of film production?",
        ]
        actual = list(
            np.array(batch["flat_questions"][0])[batch["relevant_question_indexes"][0]]
        )
        assert expected == actual, actual
        actual = list(
            np.array(batch["flat_questions"][0])[batch["selection_vector"][0]]
        )
        assert expected == actual, actual

        paraphrase_lut = batch["paraphrase_lut"][0]
        flat_questions = batch["flat_questions"][0]
        expected = [
            [
                "What is Scott Derrickson known for in the entertainment industry?",
                "What is Scott Derrickson's claim to fame in the entertainment industry?",
            ],
            [
                "What were some of the notable contributions and achievements of Edward Davis Wood Jr. in the field of filmmaking?",
                "What were some of the significant accomplishments and contributions made by Edward Davis Wood Jr. in the realm of film production?",
            ],
        ]
        for (expected_left, expected_right), (key_left, key_right) in zip(
            expected, list(paraphrase_lut.items())[::2]
        ):
            assert expected_left == flat_questions[key_left], flat_questions[key_left]
            assert expected_right == flat_questions[key_right], flat_questions[
                key_right
            ]

    # python -m unittest dataloaders.hotpot_qa_loader_test.TestHotpotQaLoader.test_no_bad_rows -v
    def test_no_bad_rows(self):
        # https://github.com/hotpotqa/hotpot/issues/47

        train_loader = get_train_loader(batch_size=2)
        val_loader = get_validation_loader(batch_size=2)
        for _ in tqdm(train_loader):
            ...

        for _ in tqdm(val_loader):
            ...


if __name__ == "__main__":
    unittest.main()
