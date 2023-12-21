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

    def _diversity_gain_sequential(
        self, question_embedding, filtered_document_embeddings
    ):
        # Aliases
        q = question_embedding
        d = filtered_document_embeddings
        merge = self.merge

        num_documents = d.shape[0]
        diversity_gain_matrix = torch.zeros((num_documents, num_documents))

        def diversity(document, document_set):
            diversity_sum = 0
            for d_prime in document_set:
                diversity_sum += 1 - F.cosine_similarity(document, d_prime)
            return diversity_sum

        for i in range(num_documents):
            for j in range(num_documents):
                q_di = merge(q, d[i])
                q_dj = merge(q, d[j])

                diversity_d_i = diversity(q_di, d)
                diversity_d_j = diversity(q_dj, d)

                diversity_gain_matrix[i][j] = diversity_d_i - diversity_d_j

        return diversity_gain_matrix

    def _diversity_gain(self, question_embedding, filtered_document_embeddings):
        # Aliases
        q = question_embedding
        d = filtered_document_embeddings
        merge = self.merge

        num_documents = d.shape[0]
        diversity_gain_matrix = torch.zeros((num_documents, num_documents))

        # Vectorized merge operation
        q_d = merge(q.repeat(num_documents, 1), d)

        # Compute cosine similarities in a vectorized way
        # Reshape q_d to (num_documents, 1, embedding_size) and d to (1, num_documents, embedding_size)
        # The resulting cosine_similarity matrix will be of shape (num_documents, num_documents)
        cosine_sim_matrix = F.cosine_similarity(q_d.unsqueeze(1), d.unsqueeze(0), dim=2)

        # Compute diversity for each pair
        diversity_matrix = 1 - cosine_sim_matrix

        # Sum over columns to get the diversity for each document after merging with the query
        diversity_per_document = diversity_matrix.sum(dim=1)

        # Compute the diversity gain matrix
        diversity_gain_matrix = diversity_per_document.unsqueeze(
            1
        ) - diversity_per_document.unsqueeze(0)

        return diversity_gain_matrix

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
        # If nothing is selected, return question_embedding alone
        if mask.sum() == 0:
            return question_embedding

        # Gather the documents that have already been selected
        filtered_document_embeddings = document_embeddings[mask]

        quality_gain_matrix = self._quality_gain(
            question_embedding, filtered_document_embeddings
        )

        diversity_gain_matrix = self._diversity_gain(
            question_embedding, filtered_document_embeddings
        )

        gain_matrix = quality_gain_matrix * diversity_gain_matrix
        scores = gain_matrix.sum(dim=-1)

        # Normalize the scores
        scores = F.softmax(scores, dim=0)

        # Compute the weighted average of the selected document embeddings
        weighted_average = torch.sum(
            filtered_document_embeddings * scores.unsqueeze(1), dim=0
        )

        # Add the question back
        weighted_average = weighted_average + question_embedding

        # Normalize the vector
        weighted_average = F.normalize(weighted_average, dim=-1)

        return weighted_average
