"""Time full softmax (chapter 01) against negative sampling (chapter 02) as the vocabulary grows.

Full softmax scores and updates every word in the vocabulary for every training pair.
Negative sampling scores and updates only 1 + K words, so it should not slow down as the
vocabulary grows. It only stays fast if drawing the K noise words is also cheap; the table
below times two ways of drawing them.

Both training loops below are copied from the two chapter scripts. We only make the vocabulary bigger.

Run:  python chapters/02-negative-sampling/speed-comparison.py
"""

import time

import numpy as np


VOCAB_SIZES = (21, 1_000, 10_000, 50_000)
D = 3
K = 3
LEARNING_RATE = 0.05
TIMED_PAIRS = 2_000  # every method trains on the same 2,000 pairs, so the times compare fairly


def make_corpus(vocab_size: int) -> list[list[str]]:
    """Made-up sentences with the same shape as the chapter's corpus: "<thing> <describing word> <group>".

    Each group has 3 things, 3 describing words and 1 group word (7 words), and every
    thing appears with every describing word, like "cats furry pets", "cats playful pets", ...
    """
    groups = max(1, vocab_size // 7)
    sentences = []
    for g in range(groups):
        for t in range(3):
            for a in range(3):
                sentences.append([f"thing{g}_{t}", f"describe{g}_{a}", f"group{g}"])
    return sentences


def make_pairs(sentences: list[list[str]], word_to_id: dict[str, int]) -> list[tuple[int, int]]:
    """Window of 1, the same as both chapters."""
    pairs = []
    for sentence in sentences:
        for i, center in enumerate(sentence):
            for j in range(max(0, i - 1), min(len(sentence), i + 2)):
                if j != i:
                    pairs.append((word_to_id[center], word_to_id[sentence[j]]))
    return pairs


def softmax(x):
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)


def sigmoid(x):
    x = np.clip(x, -15, 15)
    return 1.0 / (1.0 + np.exp(-x))


def train_full_softmax(pairs, E, W):
    """Chapter 01's loop. W has shape (D, V): one column per word."""
    for center, context in pairs:
        h = E[center].copy()
        scores = h @ W                     # V scores
        probabilities = softmax(scores)    # V probabilities
        error = probabilities.copy()
        error[context] -= 1
        gradient_W = np.outer(h, error)    # a D x V gradient: every word's column changes
        gradient_h = W @ error
        W -= LEARNING_RATE * gradient_W
        E[center] -= LEARNING_RATE * gradient_h


def train_negative_sampling(pairs, E, W, draw_negatives):
    """Chapter 02's loop. W has shape (V, D): one row per word. Only 1 + K rows change."""
    for center, context in pairs:
        h = E[center].copy()
        u = W[context].copy()
        error = sigmoid(np.dot(h, u)) - 1.0
        E[center] -= LEARNING_RATE * error * u
        W[context] -= LEARNING_RATE * error * h
        for neg in draw_negatives(context):
            h = E[center].copy()
            u_neg = W[neg].copy()
            error_neg = sigmoid(np.dot(h, u_neg))
            E[center] -= LEARNING_RATE * error_neg * u_neg
            W[neg] -= LEARNING_RATE * error_neg * h


def chapter_sampler(negative_probabilities):
    """Chapter 02's sampler: np.random.choice reads the whole probability list on every draw."""
    vocab_size = len(negative_probabilities)

    def draw(context):
        negatives = []
        while len(negatives) < K:
            word = np.random.choice(vocab_size, p=negative_probabilities)
            if word != context:
                negatives.append(word)
        return negatives
    return draw


def table_sampler(negative_probabilities, table_size=1_000_000):
    """The original word2vec trick: fill a big table once, where each word fills a share of the
    slots equal to its probability. Then a draw is just "pick a random slot", which costs the
    same for 21 words or 50,000 words."""
    table = np.random.choice(len(negative_probabilities), size=table_size, p=negative_probabilities)
    position = 0

    def draw(context):
        nonlocal position
        negatives = []
        while len(negatives) < K:
            if position == table_size:
                np.random.shuffle(table)
                position = 0
            word = table[position]
            position += 1
            if word != context:
                negatives.append(word)
        return negatives
    return draw


def timed(function, *args) -> float:
    start = time.perf_counter()
    function(*args)
    return time.perf_counter() - start


def main() -> None:
    np.random.seed(42)
    print(f"Each method trains on the same {TIMED_PAIRS:,} pairs. D = {D}, K = {K}.\n")
    print(f"{'vocabulary':>10} | {'full softmax':>12} | {'neg. sampling':>13} | {'neg. sampling':>13}")
    print(f"{'':>10} | {'(chapter 01)':>12} | {'(chapter 02)':>13} | {'(table draw)':>13}")
    print("-" * 58)
    for vocab_size in VOCAB_SIZES:
        sentences = make_corpus(vocab_size)
        vocab = sorted({word for sentence in sentences for word in sentence})
        word_to_id = {word: i for i, word in enumerate(vocab)}
        pairs = make_pairs(sentences, word_to_id)
        chosen = np.random.choice(len(pairs), size=TIMED_PAIRS, replace=True)
        pairs = [pairs[i] for i in chosen]

        counts = np.zeros(len(vocab))
        for sentence in sentences:
            for word in sentence:
                counts[word_to_id[word]] += 1
        negative_probabilities = counts ** 0.75
        negative_probabilities /= negative_probabilities.sum()

        E = np.random.randn(len(vocab), D) * 0.1
        W = np.random.randn(len(vocab), D) * 0.1
        full = timed(train_full_softmax, pairs, E.copy(), W.T.copy())
        negative = timed(train_negative_sampling, pairs, E.copy(), W.copy(), chapter_sampler(negative_probabilities))
        table = timed(train_negative_sampling, pairs, E.copy(), W.copy(), table_sampler(negative_probabilities))
        print(f"{len(vocab):>10,} | {full:>10.3f} s | {negative:>11.3f} s | {table:>11.3f} s")

    print("\nFull softmax gets slower as the vocabulary grows: every pair scores and updates all V words.")
    print("Chapter 02's sampler is slow too: np.random.choice re-reads all V probabilities on every")
    print("single draw, and each call has a fixed start-up cost. With the table draw, negative sampling")
    print("takes almost the same time for 21 words and for 50,000 words.")


if __name__ == "__main__":
    main()
