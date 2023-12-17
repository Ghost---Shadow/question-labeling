import torch


def submodular_mutual_information(question_embedding, document_embeddings, mask):
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
    scores = torch.zeros(num_docs)

    # Compute scores for each document
    for i in range(num_docs):
        if mask[i]:
            for j in range(num_docs):
                if i != j:
                    # Information Gain: Similarity between combined query+doc_i and doc_j
                    combined_query_doc = (
                        question_embedding + document_embeddings[i]
                    ) / 2

                    # TODO: Is this really info gain?
                    ig = torch.nn.functional.cosine_similarity(
                        combined_query_doc.unsqueeze(0),
                        document_embeddings[j].unsqueeze(0),
                    )

                    # Diversity Gain: Distance between doc_i and doc_j
                    # TODO: Is this really diversity gain or just diversity?
                    dg = 1 - torch.nn.functional.cosine_similarity(
                        document_embeddings[i].unsqueeze(0),
                        document_embeddings[j].unsqueeze(0),
                    )

                    # Multiply IG and DG for the score
                    scores[i] += (ig * dg).squeeze()

    # Normalize the scores
    scores = torch.nn.functional.softmax(scores[mask], dim=0)

    # Compute the weighted average of the selected document embeddings
    weighted_average = torch.sum(filtered_docs * scores.unsqueeze(1), dim=0)

    # Make sure the vector is normalized
    weighted_average = torch.nn.functional.normalize(weighted_average, dim=-1)

    return weighted_average
