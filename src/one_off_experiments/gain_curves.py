import os
from pathlib import Path
from dataloaders.hotpot_qa_loader import get_loader
from models.wrapped_mpnet import WrappedMpnetModel
import numpy as np
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt

BASE_PATH = Path("./artifacts/gain_curves")
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
    d_acc = query_embedding.clone()

    # Separate selected indices for similarities and dissimilarities
    selected_indices_sim = []
    selected_indices_dissim = []

    cosine_similarities = []
    cosine_dissimilarities = []

    for _ in tqdm(range(len(document_embeddings))):
        # For q_acc: Calculate cosine similarities
        similarities = torch.matmul(document_embeddings, q_acc.T).squeeze()
        similarities[selected_indices_sim] = -float("inf")
        max_index_sim = torch.argmax(similarities).item()
        cosine_similarities.append(similarities[max_index_sim].item())
        q_acc = (q_acc + document_embeddings[max_index_sim]) / 2
        q_acc = torch.nn.functional.normalize(q_acc, dim=-1)
        selected_indices_sim.append(max_index_sim)

        # For d_acc: Calculate cosine dissimilarities
        dissimilarities = torch.matmul(document_embeddings, d_acc.T).squeeze()
        dissimilarities[selected_indices_dissim] = float("inf")
        min_index_dissim = torch.argmin(dissimilarities).item()
        cosine_dissimilarities.append(1 - dissimilarities[min_index_dissim].item())
        d_acc = (d_acc + document_embeddings[min_index_dissim]) / 2
        d_acc = torch.nn.functional.normalize(d_acc, dim=-1)
        selected_indices_dissim.append(min_index_dissim)

    np.save(cosine_similarities_path, np.array(cosine_similarities))
    np.save(cosine_dissimilarities_path, np.array(cosine_dissimilarities))

cosine_similarities = np.load(cosine_similarities_path)
cosine_dissimilarities = np.load(cosine_dissimilarities_path)

# Plot Cosine Similarities
plt.figure(figsize=(10, 6))
for i, value in zip(selected_indices_sim, cosine_similarities):
    color = "green" if i in correct_answers else "red"
    plt.plot(i, value, "o", color=color)
plt.title("Cosine Similarities Over Iterations")
plt.xlabel("Iteration")
plt.ylabel("Cosine Similarity")
plt.grid(True)
plt.savefig(BASE_PATH / "cosine_similarities_plot.png")

# Plot Cosine Dissimilarities
plt.figure(figsize=(10, 6))
for i, value in zip(selected_indices_dissim, cosine_dissimilarities):
    color = "green" if i in correct_answers else "red"
    plt.plot(i, value, "o", color=color)
plt.title("Cosine Dissimilarities Over Iterations")
plt.xlabel("Iteration")
plt.ylabel("Cosine Dissimilarity")
plt.grid(True)
plt.savefig(BASE_PATH / "cosine_dissimilarities_plot.png")

cumulative_similarity = np.cumsum(cosine_similarities)
cumulative_dissimilarity = np.cumsum(cosine_dissimilarities)

# Normalize if the scales are very different
cumulative_similarity = cumulative_similarity / np.max(cumulative_similarity)
cumulative_dissimilarity = cumulative_dissimilarity / np.max(cumulative_dissimilarity)

# Plotting the cumulative curves
plt.figure(figsize=(12, 6))

# Cumulative Similarities
plt.subplot(1, 2, 1)
plt.plot(cumulative_similarity, marker="o", color="blue")
plt.title("Cumulative Cosine Similarities")
plt.xlabel("Number of Embeddings Added")
plt.ylabel("Cumulative Similarity")
plt.grid(True)

# Cumulative Dissimilarities
plt.subplot(1, 2, 2)
plt.plot(cumulative_dissimilarity, marker="o", color="red")
plt.title("Cumulative Cosine Dissimilarities")
plt.xlabel("Number of Embeddings Added")
plt.ylabel("Cumulative Dissimilarity")
plt.grid(True)

plt.tight_layout()
plt.savefig(BASE_PATH / "cumulative.png")
