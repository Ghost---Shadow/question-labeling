import torch


def weighted_embedding_average(question_embedding, document_embeddings, mask, weight=1):
    """
    Averages the embeddings of documents where mask is true and then averages it with the question embedding.

    Args:
    - question_embedding (torch.Tensor): The embedding of the question.
    - document_embeddings (torch.Tensor): A 2D tensor of document embeddings.
    - mask (torch.Tensor): A boolean mask indicating which document embeddings to consider.
    - weight (float): A weighing hyperparameter for averaging with the question embedding.

    Returns:
    - torch.Tensor: The weighted average embedding.
    - torch.Tensor: The mask
    """

    # If nothing is selected, then return question_embedding alone
    if mask.sum() == 0:
        return question_embedding, mask

    # Filter the document embeddings based on the mask
    filtered_docs = document_embeddings[mask]

    # Compute the mean of the filtered document embeddings
    mean_doc_embedding = torch.mean(filtered_docs, dim=0)

    # Calculate the weighted average with the question embedding
    weighted_average = (weight * question_embedding + mean_doc_embedding) / (1 + weight)

    # No operation
    new_mask = mask

    return weighted_average, new_mask
