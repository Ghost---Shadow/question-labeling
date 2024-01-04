from sentence_transformers.util import batch_to_device
import torch
from transformers import AutoTokenizer, AutoModel


class WrappedSentenceTransformerModel:
    def __init__(self, config, model=None):
        self.config = config

        checkpoint = config["architecture"]["semantic_search_model"]["checkpoint"]
        self.device = config["architecture"]["semantic_search_model"]["device"]
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        if model is None:
            self.model = AutoModel.from_pretrained(checkpoint).to(self.device)
        else:
            self.model = model

    def get_all_trainable_parameters(self):
        return self.model.parameters()

    def get_query_and_document_embeddings(self, query, documents):
        all_sentences = [query] + documents
        features = self.tokenizer(
            all_sentences, padding=True, truncation=True, return_tensors="pt"
        )
        features = batch_to_device(features, self.device)

        model_output = self.model(**features)
        all_embeddings = model_output.pooler_output

        # Extract query and document embeddings
        query_embedding = all_embeddings[0]
        document_embeddings = all_embeddings[1:]

        # normalize the vectors
        query_embedding = torch.nn.functional.normalize(query_embedding, dim=-1)
        document_embeddings = torch.nn.functional.normalize(document_embeddings, dim=-1)

        return query_embedding, document_embeddings
