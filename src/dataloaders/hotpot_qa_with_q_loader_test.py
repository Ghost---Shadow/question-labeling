import unittest
from dataloaders.hotpot_qa_with_q_loader import add_question_to_row
from datasets import load_dataset
from models.t5_model import T5ModelForQuestionGeneration


# python -m unittest dataloaders.hotpot_qa_with_q_loader_test.TestHotpotQaWithQLoader -v
class TestHotpotQaWithQLoader(unittest.TestCase):
    # python -m unittest dataloaders.hotpot_qa_with_q_loader_test.TestHotpotQaWithQLoader.test_add_question_to_row -v
    def test_add_question_to_row(self):
        dataset = load_dataset("hotpot_qa", "distractor")
        row = dataset["train"][0]

        config = {
            "architecture": {
                "question_generator_model": {
                    "name": "t5",
                    "size": "base",
                    "device": "cuda:0",
                }
            }
        }
        model = T5ModelForQuestionGeneration(config)

        modified_row = add_question_to_row(model, row)

        assert modified_row["questions"] is not None


if __name__ == "__main__":
    unittest.main()
