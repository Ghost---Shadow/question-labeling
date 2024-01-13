from sentence_transformers.util import batch_to_device
import torch
from transformers import AutoTokenizer, AutoModel


class WrappedMpnetModel:
    """
    https://huggingface.co/sentence-transformers/all-mpnet-base-v2
    """

    def __init__(self, config):
        self.config = config

        checkpoint = config["architecture"]["semantic_search_model"]["checkpoint"]
        self.device = config["architecture"]["semantic_search_model"]["device"]
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModel.from_pretrained(checkpoint).to(self.device)

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

    def get_query_and_document_embeddings_streaming(self, query, documents):
        batch_size = self.config["training"]["streaming"]["batch_size"]

        # Process the query separately
        query_features = self.tokenizer(
            query, padding=True, truncation=True, return_tensors="pt"
        )
        query_features = batch_to_device(query_features, self.device)

        # Compute and normalize the query embedding
        query_model_output = self.model(**query_features)
        query_embedding = torch.nn.functional.normalize(
            query_model_output.pooler_output, dim=-1
        )

        # Initialize list to hold document embeddings
        document_embeddings_list = []

        # Process documents in batches
        for i in range(0, len(documents), batch_size):
            # Prepare the batch
            batch_documents = documents[i : i + batch_size]
            document_features = self.tokenizer(
                batch_documents, padding=True, truncation=True, return_tensors="pt"
            )
            document_features = batch_to_device(document_features, self.device)

            # Compute and normalize document embeddings
            document_model_output = self.model(**document_features)
            document_embeddings = torch.nn.functional.normalize(
                document_model_output.pooler_output, dim=-1
            )

            # Move normalized embeddings to CPU (if necessary)
            document_embeddings_list.append(document_embeddings.cpu())

        # Concatenate all batched document embeddings
        document_embeddings = torch.cat(document_embeddings_list)

        return query_embedding.cpu(), document_embeddings

    def inner_product(self, question_embedding, document_embeddings):
        return (question_embedding @ document_embeddings.T).squeeze()

    def inner_product_streaming(self, question_embedding, document_embeddings):
        batch_size = self.config["training"]["streaming"]["batch_size"]

        # Ensure the question embedding is on the GPU
        question_embedding_gpu = question_embedding.to("cuda")

        # Initialize a list to store inner product results
        inner_product_results = []

        # Process document embeddings in batches
        for i in range(0, len(document_embeddings), batch_size):
            # Extract a batch of document embeddings
            document_batch = document_embeddings[i : i + batch_size].to("cuda")

            # Compute the inner product and keep the result on the GPU
            inner_product = (question_embedding_gpu @ document_batch.T).squeeze()

            # Store the result
            inner_product_results.append(inner_product)

        return torch.cat(inner_product_results)
