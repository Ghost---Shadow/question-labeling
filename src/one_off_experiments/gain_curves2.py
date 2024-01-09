import os
from pathlib import Path
from dataloaders.hotpot_qa_loader import get_loader
from models.wrapped_mpnet import WrappedMpnetModel
import numpy as np
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt

BASE_PATH = Path("./artifacts/gain_curves2")
BASE_PATH.mkdir(exist_ok=True, parents=True)

cosine_similarities_path = BASE_PATH / "cosine_similarities.npy"
cosine_dissimilarities_path = BASE_PATH / "cosine_dissimilarities.npy"

similarity_file_exists = os.path.isfile(cosine_similarities_path)
dissimilarity_file_exists = os.path.isfile(cosine_dissimilarities_path)

_, val_loader = get_loader(1)
i = 0
for batch in val_loader:
    if i == 5262:
        break
    i += 1

question = batch["questions"][0]
sentences = batch["flat_sentences"][0]
correct_answers = batch["relevant_sentence_indexes"][0]

print("Num correct answers", len(correct_answers))

if not similarity_file_exists or not dissimilarity_file_exists:
    config = {
        "architecture": {
            "semantic_search_model": {
                "checkpoint": "sentence-transformers/all-mpnet-base-v2",
                "device": "cuda:0",
            }
        }
    }
    model = WrappedMpnetModel(config)

    query_embedding, document_embeddings = model.get_query_and_document_embeddings(
        question, sentences
    )

    # Initialize q_acc and d_acc to query_embedding
    q_acc = query_embedding.clone()
    d_acc = torch.zeros_like(query_embedding)

    # Separate selected indices for similarities and dissimilarities
    selected_indices = []

    cosine_similarities = []
    cosine_dissimilarities = []
    combined_scores = []

    for i in tqdm(range(len(document_embeddings))):
        # For q_acc: Calculate cosine similarities
        similarities = torch.matmul(document_embeddings, q_acc.T).squeeze()
        similarities = torch.clamp(similarities, min=0, max=1)
        similarities[selected_indices] = 0

        # For d_acc: Calculate cosine dissimilarities
        # Exclude the first iteration from using d_acc
        if i > 0:
            dissimilarities = torch.matmul(document_embeddings, d_acc.T).squeeze()
        else:
            dissimilarities = torch.zeros_like(similarities)
        dissimilarities[selected_indices] = 1

        # Multiply similarities and dissimilarities and select the index
        combined_score = similarities * (1 - dissimilarities)
        selected_index = torch.argmax(combined_score).item()

        # Update lists and accumulators
        cosine_similarities.append(similarities[selected_index].item())
        cosine_dissimilarities.append(1 - dissimilarities[selected_index].item())
        combined_scores.append(combined_score[selected_index].detach().cpu().numpy())

        q_acc = (q_acc + document_embeddings[selected_index]) / 2
        q_acc = torch.nn.functional.normalize(q_acc, dim=-1)
        selected_indices.append(selected_index)

        if i > 0:
            d_acc = (d_acc + document_embeddings[selected_index]) / 2
            d_acc = torch.nn.functional.normalize(d_acc, dim=-1)

    # np.save(cosine_similarities_path, np.array(cosine_similarities))
    # np.save(cosine_dissimilarities_path, np.array(cosine_dissimilarities))

# cosine_similarities = np.load(cosine_similarities_path)
# cosine_dissimilarities = np.load(cosine_dissimilarities_path)

plt.figure(figsize=(10, 6))
for i, value in zip(selected_indices, combined_scores):
    color = "green" if i in correct_answers else "red"
    plt.plot(i, value, "o", color=color)
plt.title("Cosine Similarities Over Iterations")
plt.xlabel("Iteration")
plt.ylabel("Cosine Similarity")
plt.grid(True)
plt.savefig(BASE_PATH / "cosine_similarities_plot.png")
