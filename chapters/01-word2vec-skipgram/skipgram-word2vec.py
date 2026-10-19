from pprint import pprint
from pathlib import Path
from random import random
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.text import Text

def create_output_file() -> Path:
    """Reserve an unused numbered output file for this run."""
    script_path = Path(__file__)
    run_id = 1
    while True:
        output_path = script_path.with_name(f"{script_path.stem}.output_{run_id}.txt")
        try:
            with output_path.open("x", encoding="utf-8"):
                pass
        except FileExistsError:
            run_id += 1
        else:
            return output_path


OUTPUT_FILE = create_output_file()


def append_output(*values: object, pretty: bool = False) -> None:
    """Append console-style output to this run's numbered output file."""
    if pretty and len(values) != 1:
        raise ValueError("Pretty output requires exactly one value.")
    with OUTPUT_FILE.open("a", encoding="utf-8") as output_file:
        if pretty:
            pprint(values[0], stream=output_file)
        else:
            print(*values, file=output_file)


corpus = [
    "cats furry pets",
    "cats playful pets",
    "cats gentle pets",
    "dogs furry pets",
    "dogs playful pets",
    "dogs gentle pets",
    "rabbits furry pets",
    "rabbits playful pets",
    "rabbits gentle pets",
    "apples sweet fruit",
    "apples juicy fruit",
    "apples fresh fruit",
    "bananas sweet fruit",
    "bananas juicy fruit",
    "bananas fresh fruit",
    "oranges sweet fruit",
    "oranges juicy fruit",
    "oranges fresh fruit",
    "cars fast vehicles",
    "cars motorized vehicles",
    "cars wheeled vehicles",
    "buses fast vehicles",
    "buses motorized vehicles",
    "buses wheeled vehicles",
    "trucks fast vehicles",
    "trucks motorized vehicles",
    "trucks wheeled vehicles",
]

sentences = [sentence.lower().split() for sentence in corpus]
pprint(("sentences:", sentences))
append_output(("sentences:", sentences), pretty=True)

words = []
for sentence in sentences:
    for word in sentence:
        words.append(word.lower())

print("\nwords:", words)
append_output("\nwords:", words, pretty=True)

vocab = sorted(list(set(words)))
print("\nVocab:", vocab)
append_output("\nVocab:", vocab)

word_to_id = {}
id_to_word = {}

for index, word in enumerate(vocab):
    # enumerate returns a tuple containing the index and the word from the vocabulary
    word_to_id[word] = index
    id_to_word[index] = word

print("\nWord to ID and ID to Word mappings:")
append_output("\nWord to ID and ID to Word mappings:")
pprint(("word_to_id:", word_to_id))
append_output(("word_to_id:", word_to_id), pretty=True)
pprint(("id_to_word:", id_to_word))
append_output(("id_to_word:", id_to_word), pretty=True)


# Hyperparameters
WINDOW = 1
D = 3
LEARNING_RATE = 0.05
EPOCHS = 5
vocab_length = len(vocab)

# Generate center-context training pairs

pairs = []
for sentence in sentences:
    sentence_length = len(sentence)

    for i in range(0, sentence_length):

        center = sentence[i]
        left = max(0, i - WINDOW)
        right = min(sentence_length, i + WINDOW + 1)

        for j in range(left, right):
            if j != i:
                word = sentence[j]
                context_word_id = word_to_id[word]
                pairs.append((word_to_id[center], context_word_id))

# print pairs
print("\n Pairs:")
append_output("\n Pairs:")
for pair in pairs:
    item1 = pair[0]
    item2 = pair[1]
    print(pair, f"({id_to_word[item1]}, {id_to_word[item2]})")
    append_output(pair, f"({id_to_word[item1]}, {id_to_word[item2]})")

# random seed for numpy
np.random.seed(42)

# Embedding matrix initialization
E = []
for i in range(vocab_length):
    row = []
    for j in range(D):
        row.append(np.random.randn() * 0.1)
    E.append(row)

E = np.array(E)

print("\nE matrix:")
append_output("\nE matrix:")
pprint(E)
append_output(E, pretty=True)

# Output weight matrix initialization
W = []
for i in range(0, vocab_length):
    row = []
    for j in range(0, D):
        row.append(np.random.randn() * 0.1)
    W.append(row)

W = np.array(W).T  # Shape: (D, vocab_length).

print("\nW matrix:")
append_output("\nW matrix:")
pprint(W)
append_output(W, pretty=True)


def softmax(vec_z):
    max_z = max(vec_z)
    exponential_array = []

    # subtract the max for numerical stability
    for value in vec_z:
        exponential_array.append(math.exp(value - max_z))
        # exponential_array.append(math.exp(value))

    total = 0.0

    for value in exponential_array:
        total += value

    probabilities = []

    for value in exponential_array:
        probabilities.append(value / total)

    return probabilities

# Train Skip-gram
losses = []

# ----- Training loop -----

for epoch in range(EPOCHS):
    np.random.shuffle(pairs)
    total_loss = 0

    for center, context in pairs:

        # Forward pass
        h = E[center].copy()
        scores = h @ W
        probabilities = softmax(scores)

        # Cross-entropy loss
        loss = -np.log(probabilities[context] + 1e-12)
        total_loss += loss

        # Backpropagation
        error = probabilities.copy()
        error[context] -= 1

        gradient_W = np.outer(h, error)
        gradient_h = W @ error

        W -= LEARNING_RATE * gradient_W
        E[center] -= LEARNING_RATE * gradient_h

    losses.append(total_loss / len(pairs))
    print(f"End of epoch {epoch + 1}, Total Loss: {total_loss / len(pairs):.4f}")
    append_output(f"End of epoch {epoch + 1}, Total Loss: {total_loss / len(pairs):.4f}")

# 7. Print learned embeddings
print("\nLearned word embeddings:")
append_output("\nLearned word embeddings:")
for word, idx in word_to_id.items():
    print(word, np.round(E[idx], 3))
    append_output(word, np.round(E[idx], 3))

# 8. Plot training loss
def plot_training_loss(losses):
    plt.figure()
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Average cross-entropy loss")
    plt.title("Word2Vec Skip-gram Training")
    plt.grid(True)

plot_training_loss(losses)

# 9. Plot embeddings in 2D with PCA
def plot_embeddings(E, word_to_id):
    """Scatter the words on the first two principal components of the normalized embeddings."""
    # Project normalized embeddings so proximity reflects cosine similarity.
    normalized_embeddings = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
    centered_embeddings = normalized_embeddings - normalized_embeddings.mean(axis=0)
    _, _, principal_axes = np.linalg.svd(centered_embeddings, full_matrices=False)
    word_positions = centered_embeddings @ principal_axes[:2].T

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(word_positions[:, 0], word_positions[:, 1], s=80)
    ax.set_xlabel("Principal component 1")
    ax.set_ylabel("Principal component 2")
    ax.set_title("Word Similarity (2D PCA of Normalized Embeddings)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.margins(0.2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    # Place labels using rendered text bounds without moving the embedding points.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    label_bounds = []
    point_pixels = ax.transData.transform(word_positions)
    for word, idx in word_to_id.items():
        label = ax.annotate(
            word,
            word_positions[idx],
            xytext=(0, 0),
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=9,
            arrowprops=dict(arrowstyle="-", color="0.5", lw=0.6),
        )
        for radius in range(12, 193, 12):
            for angle in np.linspace(0, 2 * np.pi, 16, endpoint=False):
                label.set_position((radius * np.cos(angle), radius * np.sin(angle)))
                label.update_positions(renderer)
                bounds = Text.get_window_extent(label, renderer).padded(4)
                if (
                    ax.bbox.contains(bounds.x0, bounds.y0)
                    and ax.bbox.contains(bounds.x1, bounds.y1)
                    and not any(bounds.overlaps(other) for other in label_bounds)
                    and not any(bounds.padded(5).contains(x, y) for x, y in point_pixels)
                ):
                    label_bounds.append(bounds)
                    break
            else:
                continue
            break
        else:
            raise RuntimeError(f"No free plot space for label {word!r}; enlarge the figure.")

plot_embeddings(E, word_to_id)
plt.show()

def cosine_similarity(a, b):
    dot_product = np.dot(a, b)
    length_a = np.linalg.norm(a)
    length_b = np.linalg.norm(b)
    # 1e-12 avoids dividing by zero
    return dot_product / (length_a * length_b + 1e-12)


def similar_words(word, top_k=3):
    idx = word_to_id[word]
    scores = []

    for j in range(vocab_length):
        if j != idx:
            sim = cosine_similarity(E[idx], E[j])
            scores.append((id_to_word[j], sim))

    return sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]

print("Words similar to cats:", similar_words("cats"))
append_output("Words similar to cats:", similar_words("cats"))
print("Words similar to dogs:", similar_words("dogs"))
append_output("Words similar to dogs:", similar_words("dogs"))