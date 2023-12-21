import torch
import torch.nn as nn
import torch.nn.functional as F


class SubmodularMutualInformation(nn.Module):
    def __init__(self, config, model_ref):
        super(SubmodularMutualInformation, self).__init__()
        self.config = config
        self.model_ref = model_ref
        self.merge = self.model_ref.merge_model

    def _quality_gain_sequential(
        self, question_embedding, filtered_document_embeddings
    ):
        # Alias
        q = question_embedding
        d = filtered_document_embeddings
        merge = self.merge

        # Number of documents in the batch
        n = d.size(0)

        # Preparing an empty matrix to store the results
        quality_gain_matrix = torch.zeros(n, n, device=d.device)

        for i in range(n):
            for j in range(n):
                d_i = d[i]
                d_j = d[j]

                q_di = merge(q, d_i)
                q_dj = merge(q, d_j)

                q_di_dj = merge(q_di, d_j)
                q_dj_di = merge(q_dj, d_i)

                sim_q_di_dj = q @ q_di_dj.T
                sim_q_dj_di = q @ q_dj_di.T

                q_gain = sim_q_di_dj - sim_q_dj_di
                quality_gain_matrix[i, j] = q_gain

        return quality_gain_matrix

    def _quality_gain(self, question_embedding, filtered_document_embeddings):
        q = question_embedding
        d = filtered_document_embeddings
        merge = self.merge

        # Number of documents
        n = d.size(0)

        # Expand and repeat embeddings
        d_expanded = d.unsqueeze(1).expand(-1, n, -1)
        d_repeated = d.repeat(n, 1).view(n, n, -1)

        # Batch merge operations
        q_d = merge(q.expand(n, -1), d)
        q_di_dj = merge(
            q_d.unsqueeze(1).expand(-1, n, -1).reshape(n * n, -1),
            d_repeated.reshape(n * n, -1),
        ).view(n, n, -1)
        q_dj_di = merge(
            q_d.unsqueeze(0).expand(n, -1, -1).reshape(n * n, -1),
            d_expanded.reshape(n * n, -1),
        ).view(n, n, -1)

        # Compute similarities and quality gains
        sim_q_di_dj = torch.bmm(
            q.expand(n, -1).unsqueeze(1), q_di_dj.transpose(1, 2)
        ).squeeze(1)
        sim_q_dj_di = torch.bmm(
            q.expand(n, -1).unsqueeze(1), q_dj_di.transpose(1, 2)
        ).squeeze(1)

        # Quality gain matrix
        quality_gain_matrix = sim_q_di_dj - sim_q_dj_di

        return quality_gain_matrix

    def forward(self, question_embedding, document_embeddings, mask):
        """
        Compute a weighted average embedding based on submodular mutual information scores.

        Args:
        - question_embedding (torch.Tensor): The embedding of the question.
        - document_embeddings (torch.Tensor): A 2D tensor of document embeddings.
        - mask (torch.Tensor): A boolean mask indicating which document embeddings were chosen in last step.

        Returns:
        - torch.Tensor: 1D tensor of the weighted average obtained from the mask.
        """
        num_docs = document_embeddings.shape[0]

        # If nothing is selected, return question_embedding alone
        if mask.sum() == 0:
            return question_embedding

        # Filter the document embeddings based on the mask
        filtered_docs = document_embeddings[mask]

        # Initialize scores for weighted average
        scores = torch.zeros(num_docs, device=question_embedding.device)

        # Compute scores for each document
        for i in range(num_docs):
            if mask[i]:
                for j in range(num_docs):
                    if i != j:
                        # Information Gain: Similarity between combined query+doc_i and doc_j
                        combined_query_doc = (
                            question_embedding + document_embeddings[i]
                        ) / 2

                        ig = F.cosine_similarity(
                            combined_query_doc.unsqueeze(0),
                            document_embeddings[j].unsqueeze(0),
                        )

                        # Diversity Gain: Distance between doc_i and doc_j
                        dg = 1 - F.cosine_similarity(
                            document_embeddings[i].unsqueeze(0),
                            document_embeddings[j].unsqueeze(0),
                        )

                        # Multiply IG and DG for the score
                        scores[i] += (ig * dg).squeeze()

        # Normalize the scores
        scores = F.softmax(scores[mask], dim=0)

        # Compute the weighted average of the selected document embeddings
        weighted_average = torch.sum(filtered_docs * scores.unsqueeze(1), dim=0)

        # Normalize the vector
        weighted_average = F.normalize(weighted_average, dim=-1)

        return weighted_average
