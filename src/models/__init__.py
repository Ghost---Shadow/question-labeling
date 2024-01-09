from models.openai_chat_model import OpenAIChatModel
from models.t5_model import T5ModelForQuestionGeneration
from models.wrapped_mpnet import WrappedMpnetModel

MODEL_LUT = {
    "gpt-3.5-turbo": OpenAIChatModel,
    "t5": T5ModelForQuestionGeneration,
    "mpnet": WrappedMpnetModel,
}
