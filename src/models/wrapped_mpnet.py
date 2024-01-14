from models.wrapped_base_model import WrappedBaseModel


class WrappedMpnetModel(WrappedBaseModel):
    """
    https://huggingface.co/sentence-transformers/all-mpnet-base-v2
    """

    def __init__(self, config):
        super().__init__(config)

    def get_embeddings(self, features):
        model_output = self.model(**features)
        return model_output.pooler_output
