from models.wrapped_base_model import WrappedBaseModel
from transformers import AutoTokenizer, AutoModel


class WrappedDebertaModel(WrappedBaseModel):
    """
    A class to wrap the DeBERTa model for semantic search.
    https://huggingface.co/microsoft/deberta-v3-large
    """

    def __init__(self, config):
        self.config = config

        checkpoint = config["architecture"]["semantic_search_model"]["checkpoint"]
        self.device = config["architecture"]["semantic_search_model"]["device"]
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")
        self.model = AutoModel.from_pretrained(checkpoint).to(self.device)

    def get_embeddings(self, features):
        model_output = self.model(**features)
        # [batch_size, sequence_length, embedding_dim]
        # CLS_TOKEN is at 0 index
        return model_output.last_hidden_state[:, 0, :]
