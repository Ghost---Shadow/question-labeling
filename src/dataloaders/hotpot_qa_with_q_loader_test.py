import unittest
from dataloaders.hotpot_qa_with_q_loader import get_train_loader, get_validation_loader
from tqdm import tqdm
from train_utils import set_seed
import numpy as np


def row_test_inner(
    batch, question, sentences, no_paraphrase_question, paraphrased_questions
):
    expected = question
    actual = batch["questions"][0]
    assert actual == expected, actual

    expected = sentences
    actual = list(
        np.array(batch["flat_sentences"][0])[batch["relevant_sentence_indexes"][0]]
    )
    assert expected == actual, actual

    no_paraphrase_expected = no_paraphrase_question

    expected = [
        *no_paraphrase_expected,
        *paraphrased_questions,
    ]
    actual = list(
        np.array(batch["flat_questions"][0])[batch["relevant_question_indexes"][0]]
    )
    assert no_paraphrase_expected == actual, actual
    actual = list(np.array(batch["flat_questions"][0])[batch["labels_mask"][0]])
    assert expected == actual, actual

    assert (
        len(batch["relevant_question_indexes"][0]) * 2
        == np.array(batch["labels_mask"][0]).sum()
    )

    paraphrase_lut = batch["paraphrase_lut"][0]
    flat_questions = batch["flat_questions"][0]

    assert len(paraphrase_lut.keys()) == np.array(batch["labels_mask"][0]).sum()

    for idx, original, paraphrased in zip(
        batch["relevant_question_indexes"][0],
        no_paraphrase_expected,
        paraphrased_questions,
    ):
        key_left = idx
        key_right = paraphrase_lut[idx]

        assert original == flat_questions[key_left], flat_questions[key_left]
        assert paraphrased == flat_questions[key_right], flat_questions[key_right]


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
        question = "Who was born first, William March or Richard Brautigan?"
        sentences = [
            "William March (September 18, 1893 – May 15, 1954) was an American writer of psychological fiction and a highly decorated US Marine.",
            "Richard Gary Brautigan (January 30, 1935 – ca.",
        ]
        no_paraphrase_question = [
            "What were some notable achievements of William March as both a writer and a US Marine?",
            "What is the significance of Richard Brautigan's work in the literary world?",
        ]
        paraphrased_questions = [
            "What were William March's accomplishments and contributions as an American writer and US Marine?",
            "Can you rephrase the question about Richard Gary Brautigan?",
        ]
        row_test_inner(
            batch, question, sentences, no_paraphrase_question, paraphrased_questions
        )

        # Validation loader
        batch = next(iter(val_loader))
        question = "Were Scott Derrickson and Ed Wood of the same nationality?"
        sentences = [
            "Scott Derrickson (born July 16, 1966) is an American director, screenwriter and producer.",
            "Edward Davis Wood Jr. (October 10, 1924 – December 10, 1978) was an American filmmaker, actor, writer, producer, and director.",
        ]
        no_paraphrase_question = [
            "What is Scott Derrickson known for in the entertainment industry?",
            "What were some of the notable contributions and achievements of Edward Davis Wood Jr. in the field of filmmaking?",
        ]
        paraphrased_questions = [
            "What is Scott Derrickson known for?",
            "Who was Edward Davis Wood Jr.?",
        ]
        row_test_inner(
            batch, question, sentences, no_paraphrase_question, paraphrased_questions
        )

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
