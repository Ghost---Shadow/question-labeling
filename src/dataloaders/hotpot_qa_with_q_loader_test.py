import unittest
from dataloaders.hotpot_qa_with_q_loader import get_loader
from train_utils import set_seed
import numpy as np


# python -m unittest dataloaders.hotpot_qa_with_q_loader_test.TestHotpotQaWithQaLoader -v
class TestHotpotQaWithQaLoader(unittest.TestCase):
    # python -m unittest dataloaders.hotpot_qa_with_q_loader_test.TestHotpotQaWithQaLoader.test_happy_path -v
    def test_happy_path(self):
        # Set seed for deterministic testing
        set_seed(42)

        batch_size = 1

        train_loader, val_loader = get_loader(batch_size)

        # Train loader
        batch = next(iter(train_loader))
        expected = "Maurice Hines and his brother were famous for what?"
        actual = batch["questions"][0]
        assert actual == expected, actual

        expected = [
            "Hot Feet is a jukebox musical featuring the music of Earth, Wind & Fire, a book by Heru Ptah and was conceived, directed, and choreographed by Maurice Hines.",
            " He is the brother of dancer Gregory Hines.",
        ]
        actual = list(
            np.array(batch["flat_sentences"][0])[batch["relevant_sentence_indexes"][0]]
        )
        assert expected == actual, actual

        expected = [
            "What is the premise of the jukebox musical Hot Feet and who were the key contributors to its creation?",
            "Who is the brother of dancer Gregory Hines?",
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
