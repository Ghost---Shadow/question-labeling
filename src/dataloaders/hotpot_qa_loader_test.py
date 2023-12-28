import unittest
from dataloaders.hotpot_qa_with_q_loader import get_train_loader, get_validation_loader
from tqdm import tqdm
from train_utils import set_seed
import numpy as np


# python -m unittest dataloaders.hotpot_qa_loader_test.TestHotpotQaLoader -v
class TestHotpotQaLoader(unittest.TestCase):
    # python -m unittest dataloaders.hotpot_qa_loader_test.TestHotpotQaLoader.test_hotpot_qa_loader -v
    def test_hotpot_qa_loader(self):
        # Set seed for deterministic testing
        set_seed(42)

        batch_size = 1

        train_loader = get_train_loader(batch_size=batch_size)
        val_loader = get_validation_loader(batch_size=batch_size)

        # Train loader
        batch = next(iter(train_loader))
        expected = 'Who is the director of the upcoming Candian drama film in which the actress who made her film début as the love interest in Wes Anderson\'s "The Darjeeling Limited" is starring?'
        actual = batch["questions"][0]
        assert actual == expected, actual

        expected = [
            "The Death and Life of John F. Donovan is an upcoming Canadian drama film, co-written, co-produced and directed by Xavier Dolan in his English-language debut.",
            " It stars Kit Harington, Natalie Portman, Jessica Chastain, Susan Sarandon, Kathy Bates, Jacob Tremblay, Ben Schnetzer, Thandie Newton, Amara Karan, Chris Zylka, Jared Keeso, Emily Hampshire and Michael Gambon.",
            'Amara Karan (born 1984) is a Sri Lankan-English actress who made her film début as the love interest in Wes Anderson\'s "The Darjeeling Limited".',
        ]
        actual = list(
            np.array(batch["flat_sentences"][0])[batch["relevant_sentence_indexes"][0]]
        )
        assert expected == actual, actual
        assert sum(batch["selection_vector"][0]) == len(
            batch["relevant_question_indexes"][0]
        )

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

    # python -m unittest dataloaders.hotpot_qa_loader_test.TestHotpotQaLoader.test_no_bad_rows -v
    def test_no_bad_rows(self):
        # https://github.com/hotpotqa/hotpot/issues/47

        batch_size = 2
        train_loader = get_train_loader(batch_size=batch_size)
        val_loader = get_validation_loader(batch_size=batch_size)
        for _ in tqdm(train_loader):
            ...

        for _ in tqdm(val_loader):
            ...


if __name__ == "__main__":
    unittest.main()
