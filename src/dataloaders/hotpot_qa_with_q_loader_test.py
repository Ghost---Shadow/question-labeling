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
        expected = "The  Atlantic Islands of Galicia National Park and the Timanfaya National Park are the properties of what country?"
        actual = batch["questions"][0]
        assert actual == expected, actual

        expected = [
            'The Atlantic Islands of Galicia National Park (Galician: "Parque Nacional das Illas Atlánticas de Galicia" , Spanish: "Parque Nacional de las Islas Atlánticas de Galicia" ) is the only national park located in the autonomous community of Galicia, Spain.',
            'Timanfaya National Park (Spanish: "Parque Nacional de Timanfaya" ) is a Spanish national park in the southwestern part of the island of Lanzarote, Canary Islands.',
        ]
        actual = list(
            np.array(batch["flat_sentences"][0])[batch["relevant_sentence_indexes"][0]]
        )
        assert expected == actual, actual
        assert sum(batch["selection_vector"][0]) == len(
            batch["relevant_sentence_indexes"][0]
        )

        expected = [
            'The Atlantic Islands of Galicia National Park (Galician: "Parque Nacional das Illas Atlánticas de Galicia" , Spanish: "Parque Nacional de las Islas Atlánticas de Galicia" ) is the only national park located in the autonomous community of Galicia, Spain.',
            'Timanfaya National Park (Spanish: "Parque Nacional de Timanfaya" ) is a Spanish national park in the southwestern part of the island of Lanzarote, Canary Islands.',
        ]
        actual = list(
            np.array(batch["flat_questions"][0])[batch["relevant_question_indexes"][0]]
        )
        assert expected == actual, actual

        paraphrase_lut = batch["paraphrase_lut"][0]
        flat_questions = batch["flat_questions"][0]
        expected = [
            ["a", "b"],
            ["a", "b"],
        ]
        for (expected_left, expected_right), (key_left, key_right) in zip(
            expected, paraphrase_lut.items()
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
        ]
        actual = list(
            np.array(batch["flat_questions"][0])[batch["relevant_question_indexes"][0]]
        )
        assert expected == actual, actual

        paraphrase_lut = batch["paraphrase_lut"][0]
        flat_questions = batch["flat_questions"][0]
        expected = [
            ["a", "b"],
            ["a", "b"],
        ]
        for (expected_left, expected_right), (key_left, key_right) in zip(
            expected, paraphrase_lut.items()
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
