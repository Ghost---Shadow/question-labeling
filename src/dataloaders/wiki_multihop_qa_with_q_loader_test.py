import unittest
from dataloaders.hotpot_qa_with_q_loader_test import row_test_inner
from dataloaders.wiki_multihop_qa_with_q_loader import (
    get_train_loader,
    get_validation_loader,
)
from tqdm import tqdm
from train_utils import set_seed


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
        # batch = next(iter(train_loader))
        # question = "Who was born first, William March or Richard Brautigan?"
        # sentences = [
        #     "William March (September 18, 1893 – May 15, 1954) was an American writer of psychological fiction and a highly decorated US Marine.",
        #     "Richard Gary Brautigan (January 30, 1935 – ca.",
        # ]
        # no_paraphrase_question = [
        #     "What were some notable achievements of William March as both a writer and a US Marine?",
        #     "What is the significance of Richard Brautigan's work in the literary world?",
        # ]
        # paraphrased_questions = [
        #     "What were William March's accomplishments and contributions as an American writer and US Marine?",
        #     "Can you rephrase the question about Richard Gary Brautigan?",
        # ]
        # row_test_inner(
        #     batch, question, sentences, no_paraphrase_question, paraphrased_questions
        # )

        # Validation loader
        batch = next(iter(val_loader))
        question = (
            "Who is the mother of the director of film Polish-Russian War (Film)?"
        )
        sentences = [
            "(Wojna polsko-ruska) is a 2009 Polish film directed by Xawery Żuławski based on the novel Polish-Russian War under the white-red flag by Dorota Masłowska.",
            "He is the son of actress Małgorzata Braunek and director Andrzej Żuławski.",
        ]
        no_paraphrase_question = [
            'What is the plot of the film "Wojna polsko-ruska" and who directed it?',
            "Who are the parents of the person mentioned in the passage?",
        ]
        paraphrased_questions = [
            'Who directed the film "Wojna polsko-ruska" and what is its storyline?',
            "Which individuals are identified as the parents of the individual referenced in the passage?",
        ]
        row_test_inner(
            batch, question, sentences, no_paraphrase_question, paraphrased_questions
        )

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
