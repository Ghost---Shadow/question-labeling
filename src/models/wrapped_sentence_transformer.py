from sentence_transformers import SentenceTransformer
import torch


class WrappedSentenceTransformerModel:
    def __init__(self, config):
        checkpoint = config["architecture"]["semantic_search_model"]["checkpoint"]
        device = config["architecture"]["semantic_search_model"]["device"]
        self.device = device
        self.model = SentenceTransformer(checkpoint).to(device)

    def get_inner_products(self, query, documents):
        all_sentences = [query, *documents]
        all_embeddings = self.model.encode(all_sentences)

        query_embedding = torch.tensor(all_embeddings[0], device=self.device)
        query_embedding = query_embedding.unsqueeze(0)
        document_embeddings = torch.tensor(all_embeddings[1:], device=self.device)

        return (query_embedding @ document_embeddings.T).squeeze()
