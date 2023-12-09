from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()


class OpenAIChatModel:
    def __init__(self, config):
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.config = config

    def generate_question(self, passage):
        completion = self.client.chat.completions.create(
            model=self.config["architecture"]["question_generator_model"]["name"],
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": "Generate a question for the given passage.",
                },
                {"role": "user", "content": passage},
            ],
        )

        return completion.choices[0].message.content

    def generate_question_batch_lossy(self, passages):
        """
        Batch endpoint currently does not exist for chat completion.
        Grouping multiple questions in the same context risks contamination
        """
        completion = self.client.chat.completions.create(
            model=self.config["architecture"]["question_generator_model"]["name"],
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": "Generate a question each for the given passages. Deliminate with newline.",
                },
                {"role": "user", "content": "\n".join(passages)},
            ],
        )

        return completion.choices[0].message.content.split("\n")
