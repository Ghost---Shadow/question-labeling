import unittest

from models.t5_model import T5ModelForQuestionGeneration


# python -m unittest models.t5_model_test.TestT5Model -v
@unittest.skip("needs GPU")
class TestT5Model(unittest.TestCase):
    # python -m unittest models.t5_model_test.TestT5Model.test_generate_question -v
    def test_generate_question(self):
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
        result = model.generate_question("The name of the cat is Toby")
        assert result == "What is the name of the cat?"

        result = model.generate_question(
            "The name of the fox with red tail is Toby and blue tail is Rob."
        )
        assert result == "What is the name of the fox with red tail?"

        result = model.generate_question("Tom ate a banana because he was sick")
        assert result == "What did Tom do after eating the banana?"

    # python -m unittest models.t5_model_test.TestT5Model.test_generate_question_batch -v
    def test_generate_question_batch(self):
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
        result = model.generate_question_batch(
            [
                "The name of the cat is Toby",
                "The name of the fox with red tail is Toby and blue tail is Rob.",
                "Tom ate a banana because he was sick",
            ]
        )
        assert result == [
            "What is the name of the cat?",
            "What is the name of the fox with red tail?",
            "What did Tom do after eating the banana?",
        ]
