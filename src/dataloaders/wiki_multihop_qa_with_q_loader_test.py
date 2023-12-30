import unittest
from dataloaders.wiki_multihop_qa_with_q_loader import (
    get_train_loader,
    get_validation_loader,
)
from tqdm import tqdm
from train_utils import set_seed
import numpy as np


# python -m unittest dataloaders.wiki_multihop_qa_with_q_loader_test.TestWikiMultihopQaWithQLoader -v
class TestWikiMultihopQaWithQLoader(unittest.TestCase):
    # python -m unittest dataloaders.wiki_multihop_qa_with_q_loader_test.TestWikiMultihopQaWithQLoader.test_happy_path -v
    def test_happy_path(self):
        # Set seed for deterministic testing
        set_seed(42)

        batch_size = 1

        train_loader = get_train_loader(batch_size=batch_size)
        val_loader = get_validation_loader(batch_size=batch_size)

        # Train loader
        batch = next(iter(train_loader))
        expected = "Are director of film All Square and director of film The Prize Fighter both from the same country?"
        actual = batch["questions"][0]
        assert actual == expected, actual

        expected = [
            "All Square is a 2018 American drama film directed by John Hyams.",
            "Directed by Michael Preece, it was written by Tim Conway and John Myhers, based on a story by Conway.",
            'John Hyams is an American screenwriter, director and cinematographer, best known for his involvement in the" Universal Soldier" series, for which he has directed two installments.',
            'Michael Preece( born September 15, 1936) is an American film and television director, script supervisor, producer, and actor best known for directing television series" Dallas" and" Walker, Texas Ranger" and films" The Prize Fighter" and.',
        ]
        actual = list(
            np.array(batch["flat_sentences"][0])[batch["relevant_sentence_indexes"][0]]
        )
        assert expected == actual, actual
        assert sum(batch["labels_mask"][0]) == len(
            batch["relevant_question_indexes"][0]
        )

        expected = [
            'Who directed the 2018 American drama film "All Square"?',
            "Who directed the film and who wrote the screenplay for it?",
            "What is John Hyams best known for in his career as a filmmaker?",
            "What are some of Michael Preece's notable works as a director, script supervisor, producer, and actor?",
            'Which filmmaker was in charge of directing the American drama film "All Square" released in 2018?',
            "Who was the director of the film and who was responsible for writing the screenplay?",
            "What is John Hyams most notable achievement as a filmmaker?",
            "Which works by Michael Preece stand out in his career as a director, script supervisor, producer, and actor?",
        ]
        actual = list(
            np.array(batch["flat_questions"][0])[batch["relevant_question_indexes"][0]]
        )
        assert expected == actual, actual

        paraphrase_lut = batch["paraphrase_lut"][0]
        flat_questions = batch["flat_questions"][0]
        labels_mask = batch["labels_mask"][0]
        actual = list(np.array(flat_questions)[labels_mask])
        assert len(flat_questions) == len(labels_mask)
        assert set(expected) == set(actual), actual

        expected = [
            [
                'Who directed the 2018 American drama film "All Square"?',
                'Which filmmaker was in charge of directing the American drama film "All Square" released in 2018?',
            ],
            [
                "Who directed the film and who wrote the screenplay for it?",
                "Who was the director of the film and who was responsible for writing the screenplay?",
            ],
            [
                "What is John Hyams best known for in his career as a filmmaker?",
                "What is John Hyams most notable achievement as a filmmaker?",
            ],
            [
                "What are some of Michael Preece's notable works as a director, script supervisor, producer, and actor?",
                "Which works by Michael Preece stand out in his career as a director, script supervisor, producer, and actor?",
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

        expected = [
            'What is the plot of the film "Wojna polsko-ruska" and who directed it?',
            "Who are the parents of the person mentioned in the passage?",
            'Who directed the film "Wojna polsko-ruska" and what is its storyline?',
            "Which individuals are identified as the parents of the individual referenced in the passage?",
        ]
        actual = list(
            np.array(batch["flat_questions"][0])[batch["relevant_question_indexes"][0]]
        )
        assert expected == actual, actual
        actual = list(np.array(batch["flat_questions"][0])[batch["labels_mask"][0]])
        assert set(expected) == set(actual), actual

        paraphrase_lut = batch["paraphrase_lut"][0]
        flat_questions = batch["flat_questions"][0]
        expected = [
            [
                'What is the plot of the film "Wojna polsko-ruska" and who directed it?',
                'Who directed the film "Wojna polsko-ruska" and what is its storyline?',
            ],
            [
                "Who are the parents of the person mentioned in the passage?",
                "Which individuals are identified as the parents of the individual referenced in the passage?",
            ],
        ]
        for (expected_left, expected_right), (key_left, key_right) in zip(
            expected, list(paraphrase_lut.items())[::2]
        ):
            assert expected_left == flat_questions[key_left], flat_questions[key_left]
            assert expected_right == flat_questions[key_right], flat_questions[
                key_right
            ]

    # python -m unittest dataloaders.wiki_multihop_qa_with_q_loader_test.TestWikiMultihopQaWithQLoader.test_no_bad_rows -v
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
