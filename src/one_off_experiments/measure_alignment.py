from dataloaders.hotpot_qa_with_q_loader import get_loader
import argparse
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
import numpy as np


def plot_similarity_graph(model_name, data_loader, output_dir):
    model = SentenceTransformer(model_name)
    similarities = []
    perfect_alignment_counts = []

    for batch in tqdm(data_loader):
        for passages, questions in zip(
            batch["flat_sentences"], batch["flat_questions"]
        ):
            # Encode sentences & questions
            passage_embeddings = model.encode(passages, convert_to_tensor=True)
            question_embeddings = model.encode(questions, convert_to_tensor=True)

            # Compute cosine similarities and store them
            batch_similarities = util.pytorch_cos_sim(
                passage_embeddings, question_embeddings
            )
            similarity_matrix = batch_similarities.cpu().numpy()
            similarities.append(similarity_matrix)

            # Measure perfect alignment
            max_indices = np.argmax(similarity_matrix, axis=1)
            perfect_alignment = np.mean(max_indices == np.arange(len(max_indices)))
            perfect_alignment_counts.append(perfect_alignment)

    # Plotting and saving to disk
    for batch_idx, batch_similarity in enumerate(similarities):
        plt.figure(figsize=(10, 8))
        sns.heatmap(batch_similarity, annot=False)
        plt.title(f"Batch {batch_idx + 1} Similarities")
        plt.xlabel("Questions")
        plt.ylabel("Passages")

        # Save plot
        plt.savefig(f"{output_dir}/batch_{batch_idx + 1}_similarity.png")
        plt.close()

        # One is enough
        break

    # Plotting histogram of perfect alignment percentages
    plt.figure(figsize=(10, 8))
    plt.hist(perfect_alignment_counts, bins=20, edgecolor="black")
    plt.title(
        "Histogram of Perfect Alignment Fraction with GPT-3.5 turbo (Validation set)"
    )
    plt.xlabel("Perfect Alignment Fraction")
    plt.ylabel("Frequency")
    plt.savefig(f"{output_dir}/perfect_alignment_histogram.png")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        required=False,
        default="sentence-transformers/all-mpnet-base-v2",
        help="Name of the SentenceTransformer model",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=False,
        default="./artifacts",
        help="Directory to save the plots",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        required=False,
        default=1,
        help="Directory to save the plots",
    )
    args = parser.parse_args()

    batch_size = args.batch_size
    train_loader, val_loader = get_loader(batch_size)

    plot_similarity_graph(args.model_name, val_loader, args.output_dir)
