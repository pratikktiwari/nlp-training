"""GloVe with full-batch PyTorch autograd and Adagrad."""

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from numpy.typing import NDArray
import torch
from torch import nn, optim


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
FloatArray = NDArray[np.float32]


def tokenize(sentence: str) -> list[str]:
    return re.findall(r"\b\w+\b", sentence.lower())


@dataclass
class CorpusData:
    sentences: list[list[str]]
    vocab: list[str]
    word_to_id: dict[str, int]
    cooccurrence: dict[tuple[int, int], float]
    matrix: FloatArray
    target_ids: torch.Tensor
    context_ids: torch.Tensor
    counts: torch.Tensor
    corpus: list[str]
    window: int


def prepare_corpus(corpus: list[str], window: int = WINDOW) -> CorpusData:
    if window < 1:
        raise ValueError("The context window must be at least one.")
    sentences = [tokenize(sentence) for sentence in corpus]
    word_counts = Counter(word for sentence in sentences for word in sentence)
    vocab = sorted(word_counts)
    word_to_id = {word: i for i, word in enumerate(vocab)}
    cooccurrence: dict[tuple[int, int], float] = defaultdict(float)
    for sentence in sentences:
        for i, target_word in enumerate(sentence):
            target_id = word_to_id[target_word]
            for j in range(max(0, i - window), min(len(sentence), i + window + 1)):
                if i == j:
                    continue
                context_id = word_to_id[sentence[j]]
                distance = abs(i - j)
                cooccurrence[(target_id, context_id)] += 1.0 / distance
    if not cooccurrence:
        raise ValueError("The corpus must contain at least one within-sentence context pair.")
    matrix = np.zeros((len(vocab), len(vocab)), dtype=np.float32)
    for (i, j), count in cooccurrence.items():
        matrix[i, j] = count
    pairs = list(cooccurrence)
    return CorpusData(
        sentences, vocab, word_to_id, dict(cooccurrence), matrix,
        torch.tensor([i for i, _ in pairs], dtype=torch.long),
        torch.tensor([j for _, j in pairs], dtype=torch.long),
        torch.tensor([cooccurrence[pair] for pair in pairs], dtype=torch.float32),
        list(corpus), window,
    )


class GloVe(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int) -> None:
        super().__init__()
        self.target_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.target_bias = nn.Embedding(vocab_size, 1)
        self.context_bias = nn.Embedding(vocab_size, 1)
        nn.init.uniform_(self.target_embeddings.weight, -0.5 / embedding_dim, 0.5 / embedding_dim)
        nn.init.uniform_(self.context_embeddings.weight, -0.5 / embedding_dim, 0.5 / embedding_dim)
        nn.init.zeros_(self.target_bias.weight)
        nn.init.zeros_(self.context_bias.weight)

    def forward(self, target_ids: torch.Tensor, context_ids: torch.Tensor) -> torch.Tensor:
        target_vectors = self.target_embeddings(target_ids)
        context_vectors = self.context_embeddings(context_ids)
        dot_product = (target_vectors * context_vectors).sum(dim=1)
        bias_target = self.target_bias(target_ids).squeeze(1)
        bias_context = self.context_bias(context_ids).squeeze(1)
        prediction = dot_product + bias_target + bias_context
        return prediction


@dataclass
class TrainingRun:
    model: GloVe
    data: CorpusData
    parameters: list[FloatArray]
    gradients: list[FloatArray]
    accumulators: list[FloatArray]
    predictions: list[FloatArray]
    pair_losses: list[FloatArray]
    losses: list[float]
    log_counts: torch.Tensor
    weights: torch.Tensor
    optimizer_defaults: dict[str, float]


def parameter_vector(model: GloVe) -> FloatArray:
    return torch.cat([parameter.detach().reshape(-1) for parameter in model.parameters()]).cpu().numpy().copy()


def gradient_vector(model: GloVe) -> FloatArray:
    gradients = []
    for parameter in model.parameters():
        if parameter.grad is None:
            raise RuntimeError("Call loss.backward() before recording parameter gradients.")
        gradients.append(parameter.grad.detach().reshape(-1))
    return torch.cat(gradients).cpu().numpy().copy()


def accumulator_vector(model: GloVe, optimizer: optim.Adagrad) -> FloatArray:
    accumulators = []
    for parameter in model.parameters():
        value = optimizer.state[parameter]["sum"]
        if not isinstance(value, torch.Tensor):
            raise TypeError("Expected a tensor in Adagrad's accumulated squared-gradient state.")
        accumulators.append(value.detach().reshape(-1))
    return torch.cat(accumulators).cpu().numpy().copy()


def train_glove(data: CorpusData, epochs: int = EPOCHS) -> TrainingRun:
    if epochs < 1:
        raise ValueError("Train for at least one epoch.")
    torch.manual_seed(42)
    np.random.seed(42)
    model = GloVe(len(data.vocab), D).cpu()
    optimizer = optim.Adagrad(model.parameters(), lr=LEARNING_RATE)
    log_counts = torch.log(data.counts)
    weights = torch.where(
        data.counts < X_MAX,
        (data.counts / X_MAX) ** ALPHA,
        torch.ones_like(data.counts),
    )
    run = TrainingRun(
        model, data, [parameter_vector(model)], [], [accumulator_vector(model, optimizer)],
        [], [], [], log_counts, weights,
        {key: float(optimizer.defaults[key]) for key in
         ("lr", "eps", "lr_decay", "weight_decay", "initial_accumulator_value")},
    )
    for _ in range(epochs):
        model.train()
        predictions = model(data.target_ids, data.context_ids)
        loss = (weights * (predictions - log_counts) ** 2).mean()
        if not torch.isfinite(loss).item():
            raise RuntimeError("GloVe produced a non-finite training loss.")
        optimizer.zero_grad()
        loss.backward()
        run.gradients.append(gradient_vector(model))
        run.predictions.append(predictions.detach().cpu().numpy().copy())
        run.pair_losses.append((weights * (predictions - log_counts) ** 2).detach().cpu().numpy().copy())
        run.losses.append(float(loss.item()))
        optimizer.step()
        run.parameters.append(parameter_vector(model))
        run.accumulators.append(accumulator_vector(model, optimizer))
    model.eval()
    with torch.no_grad():
        predictions = model(data.target_ids, data.context_ids)
        pair_losses = weights * (predictions - log_counts) ** 2
        run.predictions.append(predictions.cpu().numpy().copy())
        run.pair_losses.append(pair_losses.cpu().numpy().copy())
        run.losses.append(float(pair_losses.mean().item()))
    return run


def final_embeddings(model: GloVe) -> FloatArray:
    with torch.no_grad():
        embeddings = model.target_embeddings.weight + model.context_embeddings.weight
    return embeddings.cpu().numpy().copy()


def word_id(word: str, data: CorpusData) -> int:
    if word not in data.word_to_id:
        raise ValueError(f"Word '{word}' is not in the vocabulary.")
    return data.word_to_id[word]


def get_embedding(word: str, embeddings: FloatArray, data: CorpusData) -> FloatArray:
    return embeddings[word_id(word, data)].copy()


def cosine_similarity(word1: str, word2: str, embeddings: FloatArray, data: CorpusData) -> float:
    v1 = get_embedding(word1, embeddings, data)
    v2 = get_embedding(word2, embeddings, data)
    denominator = np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10
    return float(np.dot(v1, v2) / denominator)


def most_similar(word: str, embeddings: FloatArray, data: CorpusData, top_k: int = 5) -> list[tuple[str, float]]:
    if not 1 <= top_k < len(data.vocab):
        raise ValueError("top_k must be between 1 and vocabulary size minus one.")
    vector = get_embedding(word, embeddings, data)
    norms = np.linalg.norm(embeddings, axis=1)
    similarities = embeddings @ vector / (norms * np.linalg.norm(vector) + 1e-10)
    similarities[word_id(word, data)] = -np.inf
    indices = np.argsort(similarities)[::-1][:top_k]
    return [(data.vocab[int(i)], round(float(similarities[i]), 4)) for i in indices]


def predict_log_count(target_word: str, context_word: str, model: GloVe, data: CorpusData) -> float:
    i, j = word_id(target_word, data), word_id(context_word, data)
    with torch.no_grad():
        prediction = model(torch.tensor([i]), torch.tensor([j]))
    return float(prediction.item())


def export_browser_run(run: TrainingRun, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    blocks = {}
    arrays = []
    offset = 0
    for name in ("parameters", "gradients", "accumulators", "predictions", "pair_losses"):
        values = np.asarray(getattr(run, name), dtype="<f4")
        blocks[name] = {"offset": offset, "frames": values.shape[0], "width": values.shape[1]}
        arrays.append(values.reshape(-1))
        offset += values.size
    binary = np.concatenate(arrays).astype("<f4").tobytes()
    source = Path(__file__).read_text(encoding="utf-8")
    layout = []
    start = 0
    for name, parameter in run.model.named_parameters():
        layout.append({"name": name, "shape": list(parameter.shape), "offset": start, "length": parameter.numel()})
        start += parameter.numel()
    prefixes = {
        "counts": ["cooccurrence[(target_id, context_id)] +="],
        "lookup": ["target_vectors = self.target_embeddings", "context_vectors = self.context_embeddings"],
        "predict": ["dot_product =", "bias_target =", "bias_context =", "prediction = dot_product"],
        "log": ["log_counts = torch.log"],
        "weight": ["weights = torch.where"],
        "loss": ["loss = (weights *"],
        "zero": ["optimizer.zero_grad()"],
        "backward": ["loss.backward()"],
        "adagrad": ["optimizer.step()"],
        "embedding": ["embeddings = model.target_embeddings.weight +"],
    }
    phase_lines = {
        phase: [n for n, line in enumerate(source.splitlines(), 1)
                if any(line.strip().startswith(prefix) for prefix in starts)]
        for phase, starts in prefixes.items()
    }
    if not all(phase_lines.values()):
        raise RuntimeError("A replay phase no longer matches the Python source. Update the export mapping.")
    manifest = {
        "schemaVersion": 1, "torchVersion": torch.__version__, "numpyVersion": np.__version__,
        "dtype": "float32", "byteOrder": "little", "binaryFile": "training.f32",
        "byteLength": len(binary), "binarySha256": hashlib.sha256(binary).hexdigest(),
        "config": {"window": run.data.window, "dimensions": run.model.target_embeddings.embedding_dim, "epochs": len(run.gradients),
                   "learningRate": LEARNING_RATE, "xMax": X_MAX, "alpha": ALPHA, "seed": 42},
        "optimizer": run.optimizer_defaults, "corpus": run.data.corpus, "sentences": run.data.sentences,
        "vocab": run.data.vocab, "pairs": [list(pair) for pair in run.data.cooccurrence],
        "matrix": run.data.matrix.tolist(), "counts": run.data.counts.tolist(),
        "logCounts": run.log_counts.tolist(), "weights": run.weights.tolist(),
        "losses": run.losses, "layout": layout, "blocks": blocks,
        "source": source, "sourceSha256": hashlib.sha256(source.encode()).hexdigest(),
        "phaseLines": phase_lines,
    }
    (destination / "training.f32").write_bytes(binary)
    (destination / "training.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-web", action="store_true", help="Regenerate the recorded PyTorch browser replay.")
    args = parser.parse_args()
    output_file = create_output_file()

    def report(*values: object) -> None:
        print(*values)
        with output_file.open("a", encoding="utf-8") as handle:
            print(*values, file=handle)

    data = prepare_corpus(CORPUS)
    report("Tokenized corpus:", data.sentences)
    report("\nVocabulary size:", len(data.vocab), "\nVocabulary:", data.vocab)
    report("\nCo-occurrence matrix shape:", data.matrix.shape, "\n", data.matrix)
    report("\nObserved directed cells:", len(data.cooccurrence))
    run = train_glove(data)
    report_epochs = sorted({epoch for epoch in (1, 2, 5, 10, 20, 50, 100, 150, 200, 250, EPOCHS) if epoch <= EPOCHS})
    for epoch in report_epochs:
        report(f"Epoch {epoch:3d}/{EPOCHS}, Loss = {run.losses[epoch - 1]:.9g} (before this update)")
    report("Loss after the final update:", f"{run.losses[-1]:.9g}")
    embeddings = final_embeddings(run.model)
    report("\nEmbedding matrix shape:", embeddings.shape)
    for word in ("ice", "steam"):
        report(f"\nEmbedding of '{word}':", get_embedding(word, embeddings, data))
    for a, b in (("ice", "water"), ("ice", "steam"), ("cats", "dogs")):
        report(f"Cosine {a} / {b}:", round(cosine_similarity(a, b, embeddings, data), 4))
    for word in ("ice", "steam", "cats"):
        report(f"Words similar to '{word}':", most_similar(word, embeddings, data))
    target, context = "ice", "water"
    report("\nPrediction inspection:", target, "/", context)
    report("Observed weighted count:", data.matrix[word_id(target, data), word_id(context, data)])
    report("Predicted log count:", predict_log_count(target, context, run.model, data))
    if args.export_web:
        destination = Path(__file__).resolve().parents[2] / "docs" / "03-glove"
        export_browser_run(run, destination)
        report("Browser replay exported to:", destination)
    report("\nGloVe training completed. Output saved to:", output_file)


if __name__ == "__main__":
    main()
