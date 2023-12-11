from sentence_transformers import SentenceTransformer
from sentence_transformers.util import batch_to_device
import torch


class WrappedSentenceTransformerModel:
    def __init__(self, config):
        checkpoint = config["architecture"]["semantic_search_model"]["checkpoint"]
        device = config["architecture"]["semantic_search_model"]["device"]
        self.device = device
        self.model = SentenceTransformer(checkpoint).to(device)

    def get_query_and_document_embeddings(self, query, documents):
        all_sentences = [query] + documents
        # Convert sentences to input format expected by the model (e.g., tokenization)
        # This depends on how SentenceTransformer expects its inputs
        features = self.model.tokenize(all_sentences)
        features = batch_to_device(features, self.device)

        # Use model.forward() to maintain the computation graph
        out_features = self.model.forward(features)
        all_embeddings = out_features["sentence_embedding"]

        # Extract query and document embeddings
        query_embedding = all_embeddings[0].unsqueeze(0)
        document_embeddings = all_embeddings[1:]

        # normalize the vectors
        query_embedding = torch.nn.functional.normalize(query_embedding, dim=-1)
        document_embeddings = torch.nn.functional.normalize(document_embeddings, dim=-1)

        return query_embedding, document_embeddings
