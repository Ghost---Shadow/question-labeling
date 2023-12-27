import unittest
from dataloaders.hotpot_qa_with_q_loader import get_loader
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

        train_loader, val_loader = get_loader(batch_size)

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

        expected = [
            "Who is the director of the upcoming Canadian drama film, The Death and Life of John F. Donovan?",
            "Who are some of the actors starring in the film mentioned in the passage?",
            "What was Amara Karan's first film role and in which movie did she make her debut?",
            "Who will be directing the Canadian drama film, The Death and Life of John F. Donovan?",
            "Which actors are mentioned in the passage as starring in the film?",
            "In which movie did Amara Karan make her first film appearance and what was her initial role?",
        ]
        actual = list(
            np.array(batch["flat_questions"][0])[batch["relevant_question_indexes"][0]]
        )
        assert expected == actual, actual

        paraphrase_lut = batch["paraphrase_lut"][0]
        flat_questions = batch["flat_questions"][0]
        expected = [
            [
                "Who is the director of the upcoming Canadian drama film, The Death and Life of John F. Donovan?",
                "Who will be directing the Canadian drama film, The Death and Life of John F. Donovan?",
            ],
            [
                "Who are some of the actors starring in the film mentioned in the passage?",
                "Which actors are mentioned in the passage as starring in the film?",
            ],
            [
                "What was Amara Karan's first film role and in which movie did she make her debut?",
                "In which movie did Amara Karan make her first film appearance and what was her initial role?",
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

        train_loader, val_loader = get_loader(batch_size=2)
        for _ in tqdm(train_loader):
            ...

        for _ in tqdm(val_loader):
            ...


if __name__ == "__main__":
    unittest.main()
