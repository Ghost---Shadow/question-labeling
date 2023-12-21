import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedEmbeddingAverage(nn.Module):
    def __init__(self, config, model_ref):
        super(WeightedEmbeddingAverage, self).__init__()
        self.config = config
        self.model_ref = model_ref
        self.weight = 1  # TODO

    def forward(self, question_embedding, document_embeddings, mask):
        """
        Averages the embeddings of documents where mask is true and then averages it with the question embedding.

        Args:
        - question_embedding (torch.Tensor): The embedding of the question.
        - document_embeddings (torch.Tensor): A 2D tensor of document embeddings.
        - mask (torch.Tensor): A boolean mask indicating which document embeddings to consider.

        Returns:
        - torch.Tensor: The weighted average embedding.
        """

        # If nothing is selected, then return question_embedding alone
        if mask.sum() == 0:
            return question_embedding

        # Filter the document embeddings based on the mask
        filtered_docs = document_embeddings[mask]

        # Compute the mean of the filtered document embeddings
        mean_doc_embedding = torch.mean(filtered_docs, dim=0)

        # Calculate the weighted average with the question embedding
        weighted_average = (self.weight * question_embedding + mean_doc_embedding) / (
            1 + self.weight
        )

        # Normalize the vector
        weighted_average = F.normalize(weighted_average, dim=-1)

        return weighted_average
