from models.openai_chat_model import OpenAIChatModel
from models.t5_model import T5ModelForQuestionGeneration


MODEL_LUT = {
    "gpt-3.5-turbo": OpenAIChatModel,
    "t5": T5ModelForQuestionGeneration,
}
