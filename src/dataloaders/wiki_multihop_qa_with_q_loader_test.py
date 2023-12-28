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
        expected = "Do both directors of films Blücher (film) and The Good Old Soak have the same nationality?"
        actual = batch["questions"][0]
        assert actual == expected, actual

        expected = [
            "The Good Old Soak is a 1937 drama film starring Wallace Beery and directed by J. Walter Ruben from a screenplay by A. E. Thomas based upon the stage play of the same name by Don Marquis.",
            "Blücher is a 1988 Norwegian thriller film directed by Oddvar Bull Tuhus, starring Helge Jordal, Frank Krog and Hege Schøyen.",
            "Jacob Walter Ruben( August 14, 1899 – September 4, 1942) was an American screenwriter, film director and producer.",
            "Oddvar Bull Tuhus( born 14 December 1940) is a Norwegian film director, script writer and television worker.",
        ]
        actual = list(
            np.array(batch["flat_sentences"][0])[batch["relevant_sentence_indexes"][0]]
        )
        assert expected == actual, actual
        assert sum(batch["selection_vector"][0]) == len(
            batch["relevant_question_indexes"][0]
        )

        expected = [
            "What is the plot of The Good Old Soak and who are the main actors and director involved in the film?",
            "What is the plot of the Norwegian thriller film Blücher?",
            "What were some notable films directed or produced by Jacob Walter Ruben?",
            "What is Oddvar Bull Tuhus known for in the entertainment industry?",
            "Can you provide a summary of the storyline in The Good Old Soak and share the names of the lead actors and director associated with the movie?",
            "Can you provide a summary of the storyline in the Norwegian thriller movie Blücher?",
            "Which films directed or produced by Jacob Walter Ruben were noteworthy?",
            "What is Oddvar Bull Tuhus famous for in the entertainment field?",
        ]
        actual = list(
            np.array(batch["flat_questions"][0])[batch["relevant_question_indexes"][0]]
        )
        assert expected == actual, actual

        paraphrase_lut = batch["paraphrase_lut"][0]
        flat_questions = batch["flat_questions"][0]
        expected = [
            [
                "What is the plot of The Good Old Soak and who are the main actors and director involved in the film?",
                "Can you provide a summary of the storyline in The Good Old Soak and share the names of the lead actors and director associated with the movie?",
            ],
            [
                "What is the plot of the Norwegian thriller film Blücher?",
                "Can you provide a summary of the storyline in the Norwegian thriller movie Blücher?",
            ],
            [
                "What were some notable films directed or produced by Jacob Walter Ruben?",
                "Which films directed or produced by Jacob Walter Ruben were noteworthy?",
            ],
            [
                "What is Oddvar Bull Tuhus known for in the entertainment industry?",
                "What is Oddvar Bull Tuhus famous for in the entertainment field?",
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
