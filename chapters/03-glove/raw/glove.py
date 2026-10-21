"""GloVe using only Python lists, hand-written gradients, and Adagrad."""

from collections import defaultdict
from dataclasses import dataclass
import math
from pathlib import Path
import random
import re


CORPUS = [
    "ice is solid",
    "ice melts in water",
    "steam is a gas",
    "water turns into steam",
    "ice and water are related",
    "steam and gas are related",
    "cats drink milk",
    "dogs drink milk",
    "cats like milk",
    "dogs chase balls",
]
WINDOW = 2
D = 20
EPOCHS = 300
LEARNING_RATE = 0.03
X_MAX = 10.0
ALPHA = 0.75
EPS = 1e-10


def tokenize(sentence: str) -> list[str]:
    return re.findall(r"\b\w+\b", sentence.lower())


@dataclass
class CorpusData:
    sentences: list[list[str]]
    vocab: list[str]
    word_to_id: dict[str, int]
    cooccurrence: dict[tuple[int, int], float]
    matrix: list[list[float]]
    target_ids: list[int]
    context_ids: list[int]
    counts: list[float]
    corpus: list[str]
    window: int


def prepare_corpus(corpus: list[str], window: int = WINDOW) -> CorpusData:
    if window < 1:
        raise ValueError("The context window must be at least one.")
    sentences = [tokenize(sentence) for sentence in corpus]
    vocab = sorted({word for sentence in sentences for word in sentence})
    word_to_id = {word: i for i, word in enumerate(vocab)}
    cooccurrence: dict[tuple[int, int], float] = defaultdict(float)
    for sentence in sentences:
        for i, target_word in enumerate(sentence):
            target_id = word_to_id[target_word]
            for j in range(max(0, i - window), min(len(sentence), i + window + 1)):
                if i == j:
                    continue
                context_id = word_to_id[sentence[j]]
                cooccurrence[(target_id, context_id)] += 1.0 / abs(i - j)
    if not cooccurrence:
        raise ValueError("The corpus must contain at least one within-sentence context pair.")
    matrix = [[0.0] * len(vocab) for _ in vocab]
    for (i, j), count in cooccurrence.items():
        matrix[i][j] = count
    return CorpusData(
        sentences, vocab, word_to_id, dict(cooccurrence), matrix,
        [i for i, _ in cooccurrence], [j for _, j in cooccurrence],
        list(cooccurrence.values()), list(corpus), window,
    )


class GloVe:
    def __init__(self, vocab_size: int, embedding_dim: int) -> None:
        if vocab_size < 1 or embedding_dim < 1:
            raise ValueError("Vocabulary size and embedding dimension must be positive.")
        rng = random.Random(42)
        bound = 0.5 / embedding_dim
        self.target_embeddings = [
            [rng.uniform(-bound, bound) for _ in range(embedding_dim)]
            for _ in range(vocab_size)
        ]
        self.context_embeddings = [
            [rng.uniform(-bound, bound) for _ in range(embedding_dim)]
            for _ in range(vocab_size)
        ]
        self.target_bias = [[0.0] for _ in range(vocab_size)]
        self.context_bias = [[0.0] for _ in range(vocab_size)]

    def parameter_rows(self) -> list[list[float]]:
        return self.target_embeddings + self.context_embeddings + self.target_bias + self.context_bias

    def forward(self, target_ids: list[int], context_ids: list[int]) -> list[float]:
        predictions = []
        for i, j in zip(target_ids, context_ids, strict=True):
            dot_product = sum(
                w * u for w, u in zip(self.target_embeddings[i], self.context_embeddings[j], strict=True)
            )
            predictions.append(dot_product + self.target_bias[i][0] + self.context_bias[j][0])
        return predictions


@dataclass
class TrainingRun:
    model: GloVe
    data: CorpusData
    parameters: list[list[float]]
    gradients: list[list[float]]
    accumulators: list[list[float]]
    predictions: list[list[float]]
    pair_losses: list[list[float]]
    losses: list[float]
    log_counts: list[float]
    weights: list[float]


def parameter_vector(model: GloVe) -> list[float]:
    return [value for row in model.parameter_rows() for value in row]


def loss_and_gradients(
    model: GloVe, data: CorpusData, log_counts: list[float], weights: list[float],
) -> tuple[list[float], list[float], list[float]]:
    predictions = model.forward(data.target_ids, data.context_ids)
    pair_losses = []
    v, d = len(model.target_embeddings), len(model.target_embeddings[0])
    gradients = [0.0] * (2 * v * d + 2 * v)
    for pair, (i, j) in enumerate(zip(data.target_ids, data.context_ids, strict=True)):
        residual = predictions[pair] - log_counts[pair]
        pair_losses.append(weights[pair] * residual * residual)
        slope = 2.0 * weights[pair] * residual / len(data.counts)
        for k in range(d):
            gradients[i * d + k] += slope * model.context_embeddings[j][k]
            gradients[v * d + j * d + k] += slope * model.target_embeddings[i][k]
        gradients[2 * v * d + i] += slope
        gradients[2 * v * d + v + j] += slope
    return predictions, pair_losses, gradients


def train_glove(data: CorpusData, epochs: int = EPOCHS) -> TrainingRun:
    if epochs < 1:
        raise ValueError("Train for at least one epoch.")
    model = GloVe(len(data.vocab), D)
    log_counts = [math.log(count) for count in data.counts]
    weights = [(count / X_MAX) ** ALPHA if count < X_MAX else 1.0 for count in data.counts]
    accumulator = [0.0] * len(parameter_vector(model))
    run = TrainingRun(
        model, data, [parameter_vector(model)], [], [accumulator.copy()],
        [], [], [], log_counts, weights,
    )
    for epoch in range(epochs + 1):
        predictions, pair_losses, gradients = loss_and_gradients(model, data, log_counts, weights)
        loss = sum(pair_losses) / len(pair_losses)
        if not math.isfinite(loss):
            raise RuntimeError("GloVe produced a non-finite training loss.")
        run.predictions.append(predictions)
        run.pair_losses.append(pair_losses)
        run.losses.append(loss)
        if epoch == epochs:
            break
        run.gradients.append(gradients)
        # Accumulate the entire batch before changing any shared vector or bias.
        index = 0
        for row in model.parameter_rows():
            for k in range(len(row)):
                gradient = gradients[index]
                accumulator[index] += gradient * gradient
                row[k] -= LEARNING_RATE * gradient / (math.sqrt(accumulator[index]) + EPS)
                index += 1
        run.parameters.append(parameter_vector(model))
        run.accumulators.append(accumulator.copy())
    return run


def final_embeddings(model: GloVe) -> list[list[float]]:
    return [
        [w + u for w, u in zip(target, context, strict=True)]
        for target, context in zip(model.target_embeddings, model.context_embeddings, strict=True)
    ]


def word_id(word: str, data: CorpusData) -> int:
    if word not in data.word_to_id:
        raise ValueError(f"Word '{word}' is not in the vocabulary.")
    return data.word_to_id[word]


def get_embedding(word: str, embeddings: list[list[float]], data: CorpusData) -> list[float]:
    return embeddings[word_id(word, data)].copy()


def cosine_similarity(
    word1: str, word2: str, embeddings: list[list[float]], data: CorpusData,
) -> float:
    v1 = get_embedding(word1, embeddings, data)
    v2 = get_embedding(word2, embeddings, data)
    denominator = math.sqrt(sum(x * x for x in v1)) * math.sqrt(sum(x * x for x in v2)) + EPS
    return sum(a * b for a, b in zip(v1, v2, strict=True)) / denominator


def most_similar(
    word: str, embeddings: list[list[float]], data: CorpusData, top_k: int = 5,
) -> list[tuple[str, float]]:
    if not 1 <= top_k < len(data.vocab):
        raise ValueError("top_k must be between 1 and vocabulary size minus one.")
    word_id(word, data)
    scores = [
        (other, cosine_similarity(word, other, embeddings, data))
        for other in data.vocab if other != word
    ]
    scores.sort(key=lambda item: item[1], reverse=True)
    return [(other, round(score, 4)) for other, score in scores[:top_k]]


def predict_log_count(target_word: str, context_word: str, model: GloVe, data: CorpusData) -> float:
    i, j = word_id(target_word, data), word_id(context_word, data)
    return model.forward([i], [j])[0]


def create_output_file() -> Path:
    script_path = Path(__file__)
    run_id = 1
    while True:
        path = script_path.with_name(f"{script_path.stem}.output_{run_id}.txt")
        try:
            with path.open("x", encoding="utf-8"):
                pass
        except FileExistsError:
            run_id += 1
        else:
            return path


def main() -> None:
    output_file = create_output_file()

    def report(*values: object) -> None:
        print(*values)
        with output_file.open("a", encoding="utf-8") as handle:
            print(*values, file=handle)

    data = prepare_corpus(CORPUS)
    report("Tokenized corpus:", data.sentences)
    report("\nVocabulary size:", len(data.vocab), "\nVocabulary:", data.vocab)
    report("\nCo-occurrence matrix shape:", (len(data.vocab), len(data.vocab)))
    for row in data.matrix:
        report(row)
    report("\nObserved directed cells:", len(data.cooccurrence))
    run = train_glove(data)
    report_epochs = sorted({epoch for epoch in (1, 2, 5, 10, 20, 50, 100, 150, 200, 250, EPOCHS) if epoch <= EPOCHS})
    for epoch in report_epochs:
        report(f"Epoch {epoch:3d}/{EPOCHS}, Loss = {run.losses[epoch - 1]:.9g} (before this update)")
    report("Loss after the final update:", f"{run.losses[-1]:.9g}")
    embeddings = final_embeddings(run.model)
    report("\nEmbedding matrix shape:", (len(embeddings), len(embeddings[0])))
    for word in ("ice", "steam"):
        report(f"\nEmbedding of '{word}':", get_embedding(word, embeddings, data))
    for a, b in (("ice", "water"), ("ice", "steam"), ("cats", "dogs")):
        report(f"Cosine {a} / {b}:", round(cosine_similarity(a, b, embeddings, data), 4))
    for word in ("ice", "steam", "cats"):
        report(f"Words similar to '{word}':", most_similar(word, embeddings, data))
    target, context = "ice", "water"
    report("\nPrediction inspection:", target, "/", context)
    report("Observed weighted count:", data.matrix[word_id(target, data)][word_id(context, data)])
    report("Predicted log count:", predict_log_count(target, context, run.model, data))
    report("\nGloVe training completed. Output saved to:", output_file)


if __name__ == "__main__":
    main()
