from pathlib import Path
from dataloaders.hotpot_qa_loader import get_loader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

_, val_loader = get_loader(1)

lengths = []
longest_index = -1
max_length = -1

for i, batch in tqdm(enumerate(val_loader)):
    correct_answers = batch["relevant_sentence_indexes"][0]
    length = len(correct_answers)
    lengths.append(length)

    if length > max_length:
        max_length = length
        longest_index = i

# Convert lengths to a numpy array for analysis
lengths_array = np.array(lengths)

print(longest_index)

# Create a histogram of the lengths
plt.hist(
    lengths_array,
    bins=range(np.min(lengths_array), np.max(lengths_array) + 2),
    align="left",
    color="blue",
    rwidth=0.8,
)
plt.title("Histogram of Lengths of Correct Answers")
plt.xlabel("Length")
plt.ylabel("Frequency")
plt.grid(True)
BASE_PATH = Path("./artifacts/gain_curves")
plt.savefig(BASE_PATH / "len_correct_answers.png")
