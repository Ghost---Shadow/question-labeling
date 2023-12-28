import unittest
from dataloaders.wiki_multihop_qa_loader import get_train_loader, get_validation_loader
from tqdm import tqdm
from train_utils import set_seed
import numpy as np


# python -m unittest dataloaders.wiki_multihop_qa_loader_test.TestWikiMultihopQaLoader -v
class TestWikiMultihopQaLoader(unittest.TestCase):
    # python -m unittest dataloaders.wiki_multihop_qa_loader_test.TestWikiMultihopQaLoader.test_happy_path -v
    def test_happy_path(self):
        # Set seed for deterministic testing
        set_seed(42)

        batch_size = 1

        train_loader = get_train_loader(batch_size=batch_size)
        val_loader = get_validation_loader(batch_size=batch_size)

        # Train loader
        batch = next(iter(train_loader))
        expected = (
            "Are both Hanover Insurance and Broadway Video located in the same country?"
        )
        actual = batch["questions"][0]
        assert actual == expected, actual

        expected = [
            "The Hanover Insurance Group, Inc., based in Worcester, Massachusetts, is one of the oldest continuous businesses in the United States still operating within its original industry.",
            'Broadway Video is an American multimedia entertainment studio founded by Lorne Michaels, creator of the sketch comedy TV series" Saturday Night Live" and producer of other television programs and movies.',
        ]
        actual = list(
            np.array(batch["flat_sentences"][0])[batch["relevant_sentence_indexes"][0]]
        )
        assert expected == actual, actual
        assert sum(batch["selection_vector"][0]) == len(
            batch["relevant_sentence_indexes"][0]
        )

        # Validation loader
        batch = next(iter(val_loader))
        expected = (
            "Who is the mother of the director of film Polish-Russian War (Film)?"
        )
        actual = batch["questions"][0]
        assert actual == expected, actual

        expected = [
            "(Wojna polsko-ruska) is a 2009 Polish film directed by Xawery Żuławski based on the novel Polish-Russian War under the white-red flag by Dorota Masłowska.",
            "He is the son of actress Małgorzata Braunek and director Andrzej Żuławski.",
        ]
        actual = list(
            np.array(batch["flat_sentences"][0])[batch["relevant_sentence_indexes"][0]]
        )
        assert expected == actual, actual

        assert expected == actual, actual

    # python -m unittest dataloaders.wiki_multihop_qa_loader_test.TestWikiMultihopQaLoader.test_no_bad_rows -v
    def test_no_bad_rows(self):
        batch_size = 2
        train_loader = get_train_loader(batch_size=batch_size)
        val_loader = get_validation_loader(batch_size=batch_size)

        for _ in tqdm(train_loader):
            ...

        for _ in tqdm(val_loader):
            ...


if __name__ == "__main__":
    unittest.main()
