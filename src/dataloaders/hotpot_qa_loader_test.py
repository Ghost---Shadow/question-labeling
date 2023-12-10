import unittest
from dataloaders.hotpot_qa_loader import get_loader
from train_utils import set_seed
import numpy as np


# python -m unittest dataloaders.hotpot_qa_loader_test.TestHotpotQaLoader -v
class TestHotpotQaLoader(unittest.TestCase):
    # python -m unittest dataloaders.hotpot_qa_loader_test.TestHotpotQaLoader.test_hotpot_qa_loader -v
    def test_hotpot_qa_loader(self):
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


if __name__ == "__main__":
    unittest.main()
