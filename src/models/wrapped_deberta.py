from models.wrapped_base_model import WrappedBaseModel


class WrappedDebertaModel(WrappedBaseModel):
    """
    A class to wrap the DeBERTa model for semantic search.
    https://huggingface.co/microsoft/deberta-v3-large
    """

    def __init__(self, config):
        super().__init__(config)

    def get_embeddings(self, features):
        model_output = self.model(**features)
        # [batch_size, sequence_length, embedding_dim]
        # CLS_TOKEN is at 0 index
        return model_output.last_hidden_state[:, 0, :]
