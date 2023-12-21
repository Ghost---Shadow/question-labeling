from sentence_transformers import SentenceTransformer
from sentence_transformers.util import batch_to_device
import torch
from merge_functions import MERGE_STRATEGY_LUT
from pydash import get


class WrappedSentenceTransformerModel:
    def __init__(self, config):
        checkpoint = config["architecture"]["semantic_search_model"]["checkpoint"]
        device = config["architecture"]["semantic_search_model"]["device"]
        merge_strategy_name = get(
            config, "architecture.aggregation_strategy.merge_strategy.name", None
        )

        MergeModelClass = MERGE_STRATEGY_LUT[merge_strategy_name]

        self.device = device
        self.model = SentenceTransformer(checkpoint).to(device)

        # Set output dim
        output_dim = self.model.get_sentence_embedding_dimension()
        config["architecture"]["semantic_search_model"]["output_dim"] = output_dim

        self.merge_model = MergeModelClass(config)

    def get_all_trainable_parameters(self):
        return [*list(self.model.parameters()), *list(self.merge_model.parameters())]

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
        query_embedding = all_embeddings[0]
        document_embeddings = all_embeddings[1:]

        # normalize the vectors
        query_embedding = torch.nn.functional.normalize(query_embedding, dim=-1)
        document_embeddings = torch.nn.functional.normalize(document_embeddings, dim=-1)

        return query_embedding, document_embeddings
