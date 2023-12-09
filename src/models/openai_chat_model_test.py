import unittest
from models.openai_chat_model import OpenAIChatModel


# python -m unittest models.openai_chat_model_test.TestOpenAiChatModel -v
# @unittest.skip("needs api key")
class TestOpenAiChatModel(unittest.TestCase):
    # python -m unittest models.openai_chat_model_test.TestOpenAiChatModel.test_generate_question -v
    def test_generate_question(self):
        config = {
            "architecture": {"question_generator_model": {"name": "gpt-3.5-turbo"}}
        }
        model = OpenAIChatModel(config)
        result = model.generate_question("The name of the cat is Toby")
        print(result)  # What is the name of the cat mentioned in the passage?

        result = model.generate_question(
            "The name of the fox with red tail is Toby and blue tail is Rob."
        )
        print(result)  # What are the names of the foxes with red and blue tails?

        result = model.generate_question("Tom ate a banana because he was sick")
        print(result)  # Why did Tom eat a banana?

    # python -m unittest models.openai_chat_model_test.TestOpenAiChatModel.test_generate_question_batch_lossy -v
    def test_generate_question_batch_lossy(self):
        config = {
            "architecture": {"question_generator_model": {"name": "gpt-3.5-turbo"}}
        }
        model = OpenAIChatModel(config)
        result = model.generate_question_batch_lossy(
            [
                "The name of the cat is Toby",
                "The name of the fox with red tail is Toby and blue tail is Rob.",
                "Tom ate a banana because he was sick",
            ]
        )
        print(result)
        # ['What is the name of the cat?', 'What are the names of the foxes with red and blue tails?', 'Why did Tom eat a banana?']
