"""Count and neural n-gram language models using only the Python standard library.

Run without arguments for a small demo, or use --wikitext for WikiText-2.
Full WikiText-2 neural training is extremely slow with Python loops.
"""

import argparse
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass
import math
from pathlib import Path
import random
import time
import urllib.request


CHAPTER_DIR = Path(__file__).resolve().parent
DATA_DIR = CHAPTER_DIR.parent / "data"
DATA_URL = "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/{split}.txt"
CORPUS = [
    "the cat sat on the mat",
    "the dog sat on the rug",
    "the cat ate the fish",
    "the dog ate the bone",
    "a cat sat on a chair",
    "a dog ran in the park",
    "the bird sat on the tree",
    "the bird ate the seed",
    "a cat ran to the door",
    "the dog ran to the door",
]
VALID_CORPUS = ["the cat sat on the rug", "a dog ate the fish"]
TEST_CORPUS = ["the dog sat on a chair", "the bird ran to the park"]
N_VALUES = (2, 3, 4)
PROMPTS = ["the cat sat", "the dog sat", "the bird ate"]
SEED = 42
Context = tuple[int, ...]
CountModel = list[dict[Context, Counter[int]]]
Reporter = Callable[[str], None]


def download_wikitext() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for split in ("train", "valid", "test"):
        path = DATA_DIR / f"wiki.{split}.txt"
        if not path.exists():
            print(f"Downloading {split} split to {path} ...")
            with urllib.request.urlopen(DATA_URL.format(split=split), timeout=60) as response:
                contents = response.read()
            path.write_bytes(contents)


def tokenize_lines(lines: list[str]) -> list[str]:
    tokens = []
    for line in lines:
        line = line.strip()
        if line and not line.startswith("="):
            tokens.extend(line.lower().split())
            tokens.append("<eos>")
    return tokens


def read_tokens(path: Path) -> list[str]:
    return tokenize_lines(path.read_text(encoding="utf-8").splitlines())


def build_vocab(train_tokens: list[str], size: int) -> list[str]:
    if size < 2:
        raise ValueError("Vocabulary size must be at least two, including <unk>.")
    counts = Counter(train_tokens)
    counts.pop("<unk>", None)
    if not counts:
        raise ValueError("Training text must contain words besides <unk>.")
    return [word for word, _ in counts.most_common(size - 1)] + ["<unk>"]


def to_ids(tokens: list[str], word_to_id: dict[str, int]) -> list[int]:
    return [word_to_id.get(word, word_to_id["<unk>"]) for word in tokens]


def make_windows(ids: list[int], n: int) -> tuple[list[Context], list[int]]:
    if n < 2 or len(ids) < n:
        raise ValueError("Use n >= 2 and at least n tokens in every split.")
    return [tuple(ids[i:i + n - 1]) for i in range(len(ids) - n + 1)], ids[n - 1:]


def train_count_model(contexts: list[Context], targets: list[int]) -> CountModel:
    if not contexts:
        raise ValueError("Count training needs at least one window.")
    model: CountModel = []
    context_length = len(contexts[0])
    for k in range(context_length + 1):
        counts: dict[Context, Counter[int]] = defaultdict(Counter)
        for context, target in zip(contexts, targets, strict=True):
            counts[context[context_length - k:]][target] += 1
        model.append(dict(counts))
    return model


def count_lookup(model: CountModel, context: Context) -> tuple[Counter[int], int]:
    if len(context) != len(model) - 1:
        raise ValueError("Context length must match the count model.")
    for k in range(len(context), -1, -1):
        counts = model[k].get(context[len(context) - k:])
        if counts:
            return counts, k
    raise RuntimeError("The count model has no empty-context fallback.")


def count_predict(model: CountModel, context: Context) -> tuple[int, int]:
    counts, used = count_lookup(model, context)
    return counts.most_common(1)[0][0], used


def count_accuracy(
    model: CountModel, contexts: list[Context], targets: list[int],
) -> tuple[float, float]:
    if not targets:
        raise ValueError("Evaluation needs at least one window.")
    correct = full_context_seen = 0
    for context, target in zip(contexts, targets, strict=True):
        prediction, used = count_predict(model, context)
        correct += prediction == target
        full_context_seen += used == len(context)
    return correct / len(targets), full_context_seen / len(targets)


def count_probabilities(model: CountModel, context: Context, vocab_size: int, unk_id: int) -> list[float]:
    if len(context) != len(model) - 1:
        raise ValueError("Context length must match the count model.")
    for k in range(len(context), -1, -1):
        counts = model[k].get(context[len(context) - k:], Counter())
        total = sum(count for word, count in counts.items() if word != unk_id)
        if total:
            return [counts.get(word, 0) / total if word != unk_id else 0.0 for word in range(vocab_size)]
    raise RuntimeError("The training text has no words besides <unk>.")


@dataclass
class NeuralModel:
    E: list[list[float]]
    W: list[list[float]]
    b: list[float]
    context_length: int


def init_neural_model(vocab_size: int, n: int, dimensions: int, rng: random.Random) -> NeuralModel:
    if vocab_size < 2 or n < 2 or dimensions < 1:
        raise ValueError("Use vocabulary size >= 2, n >= 2, and positive embedding dimensions.")
    return NeuralModel(
        [[rng.gauss(0, 0.1) for _ in range(dimensions)] for _ in range(vocab_size)],
        [[rng.gauss(0, 0.1) for _ in range(vocab_size)] for _ in range((n - 1) * dimensions)],
        [0.0] * vocab_size, n - 1,
    )


def forward(model: NeuralModel, context: Context) -> tuple[list[float], list[float]]:
    if len(context) != model.context_length:
        raise ValueError("Context length must match the neural model.")
    x = [value for word in context for value in model.E[word]]
    scores = model.b.copy()
    for value, row in zip(x, model.W, strict=True):
        for word, weight in enumerate(row):
            scores[word] += value * weight
    return x, scores


def softmax(scores: list[float]) -> list[float]:
    if not scores or not all(math.isfinite(score) for score in scores):
        raise ValueError("Softmax requires nonempty, finite scores.")
    largest = max(scores)
    exponentials = [math.exp(score - largest) for score in scores]
    total = sum(exponentials)
    return [value / total for value in exponentials]


def cross_entropy(scores: list[float], target: int) -> float:
    largest = max(scores)
    return math.log(sum(math.exp(score - largest) for score in scores)) + (largest - scores[target])


def train_step(
    model: NeuralModel, contexts: list[Context], targets: list[int], learning_rate: float,
) -> float:
    if not targets or not math.isfinite(learning_rate) or learning_rate < 0:
        raise ValueError("Use a nonempty batch and a finite, nonnegative learning rate.")
    dimensions, vocab_size = len(model.E[0]), len(model.b)
    d_W = [[0.0] * vocab_size for _ in model.W]
    d_b = [0.0] * vocab_size
    d_E: dict[int, list[float]] = {}
    total_loss = 0.0
    for context, target in zip(contexts, targets, strict=True):
        x, scores = forward(model, context)
        d_scores = softmax(scores)
        total_loss += cross_entropy(scores, target)
        d_scores[target] -= 1.0
        d_scores = [gradient / len(targets) for gradient in d_scores]
        for word, gradient in enumerate(d_scores):
            d_b[word] += gradient
        for coordinate, (value, row) in enumerate(zip(x, model.W, strict=True)):
            d_x = 0.0
            for word, gradient in enumerate(d_scores):
                d_W[coordinate][word] += value * gradient
                d_x += row[word] * gradient
            word_id = context[coordinate // dimensions]
            if word_id not in d_E:
                d_E[word_id] = [0.0] * dimensions
            d_E[word_id][coordinate % dimensions] += d_x
    # All examples use pre-update weights; repeated word IDs accumulate into one row.
    for row, gradient_row in zip(model.W, d_W, strict=True):
        for word, gradient in enumerate(gradient_row):
            row[word] -= learning_rate * gradient
    for word, gradient in enumerate(d_b):
        model.b[word] -= learning_rate * gradient
    for word, gradients in d_E.items():
        for coordinate, gradient in enumerate(gradients):
            model.E[word][coordinate] -= learning_rate * gradient
    return total_loss / len(targets)


def neural_evaluate(
    model: NeuralModel, contexts: list[Context], targets: list[int],
) -> tuple[float, float]:
    if not targets:
        raise ValueError("Evaluation needs at least one window.")
    correct, total_loss = 0, 0.0
    for context, target in zip(contexts, targets, strict=True):
        _, scores = forward(model, context)
        softmax(scores)
        correct += max(range(len(scores)), key=scores.__getitem__) == target
        total_loss += cross_entropy(scores, target)
    return correct / len(targets), total_loss / len(targets)


def train_neural_model(
    contexts: list[Context], targets: list[int], valid_contexts: list[Context],
    valid_targets: list[int], vocab_size: int, n: int, dimensions: int,
    epochs: int, batch_size: int, learning_rate: float, report: Reporter,
) -> NeuralModel:
    if epochs < 1 or batch_size < 1 or not targets:
        raise ValueError("Epochs, batch size, and training window count must be positive.")
    rng = random.Random(SEED)
    model = init_neural_model(vocab_size, n, dimensions, rng)
    order = list(range(len(targets)))
    for epoch in range(epochs):
        rng.shuffle(order)
        total_loss = 0.0
        for start in range(0, len(order), batch_size):
            batch = order[start:start + batch_size]
            loss = train_step(
                model, [contexts[i] for i in batch], [targets[i] for i in batch], learning_rate,
            )
            total_loss += loss * len(batch)
        accuracy, loss = neural_evaluate(model, valid_contexts, valid_targets)
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == epochs:
            report(f"  epoch {epoch + 1}/{epochs}: train loss {total_loss / len(targets):.3f}, "
                   f"validation loss {loss:.3f}, validation accuracy {accuracy:.2%}")
    return model


def neural_probabilities(model: NeuralModel, context: Context, unk_id: int) -> list[float]:
    _, scores = forward(model, context)
    probabilities = softmax([score for word, score in enumerate(scores) if word != unk_id])
    probabilities.insert(unk_id, 0.0)
    return probabilities


def generate(
    next_word_probabilities: Callable[[Context], list[float]], prompt_ids: list[int],
    context_length: int, words: int, rng: random.Random | None,
) -> list[int]:
    if context_length < 1 or len(prompt_ids) < context_length or words < 0:
        raise ValueError("Provide enough prompt words for the context and a nonnegative generation length.")
    text = list(prompt_ids)
    for _ in range(words):
        probabilities = next_word_probabilities(tuple(text[-context_length:]))
        if rng is None:
            word = max(range(len(probabilities)), key=probabilities.__getitem__)
        else:
            word = rng.choices(range(len(probabilities)), weights=probabilities, k=1)[0]
        text.append(word)
    return text


def detokenize(words: list[str]) -> str:
    text = (" ".join(words) + " ").replace(" <eos> ", "\n\n").replace(" @-@ ", "-").replace(" @,@ ", ",").replace(" @.@ ", ".").strip()
    for mark in (" ,", " .", " ;", " :", " 's", " )", " n't"):
        text = text.replace(mark, mark.strip() if mark[1] in ",.;:)" else mark[1:])
    return text.replace("( ", "(")


def create_output_file() -> Path:
    run_id = 1
    while True:
        path = CHAPTER_DIR / f"ngram-language-model.output_{run_id}.txt"
        try:
            with path.open("x", encoding="utf-8"):
                pass
        except FileExistsError:
            run_id += 1
        else:
            return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wikitext", action="store_true", help="Use WikiText-2 instead of the built-in demo.")
    parser.add_argument("--vocab-size", type=int, help="Vocabulary cap including <unk> (demo: 100, WikiText: 10000).")
    parser.add_argument("--dimensions", type=int, help="Embedding dimensions (demo: 8, WikiText: 64).")
    parser.add_argument("--epochs", type=int, help="Training epochs (demo: 100, WikiText: 1).")
    parser.add_argument("--batch-size", type=int, help="Batch size (demo: 8, WikiText: 512).")
    parser.add_argument("--learning-rate", type=float, help="Learning rate (demo: 0.5, WikiText: 4.0).")
    parser.add_argument("--max-tokens", type=int, help="Use only the first this many tokens of each split.")
    parser.add_argument("--generate-words", type=int, default=60, help="Words generated after each prompt (default: 60).")
    args = parser.parse_args()
    defaults = (10_000, 64, 1, 512, 4.0) if args.wikitext else (100, 8, 100, 8, 0.5)
    vocab_size = defaults[0] if args.vocab_size is None else args.vocab_size
    dimensions = defaults[1] if args.dimensions is None else args.dimensions
    epochs = defaults[2] if args.epochs is None else args.epochs
    batch_size = defaults[3] if args.batch_size is None else args.batch_size
    learning_rate = defaults[4] if args.learning_rate is None else args.learning_rate
    if any(value is not None and value < minimum for value, minimum in (
        (args.vocab_size, 2), (args.dimensions, 1), (args.epochs, 1),
        (args.batch_size, 1), (args.max_tokens, max(N_VALUES)), (args.generate_words, 0),
    )):
        parser.error("Use vocab size >= 2, positive dimensions/epochs/batch size, max tokens >= 4, and generation length >= 0.")
    if not math.isfinite(learning_rate) or learning_rate < 0:
        parser.error("Learning rate must be finite and nonnegative.")
    if args.wikitext:
        print("Warning: full WikiText-2 neural training is extremely slow in pure Python. "
              "Use --max-tokens, --vocab-size, and --dimensions to reduce the workload.", flush=True)
        download_wikitext()
        splits = [read_tokens(DATA_DIR / f"wiki.{split}.txt") for split in ("train", "valid", "test")]
        prompts = ["the game was", "in the first season of the", "the city is located"]
    else:
        splits = [tokenize_lines(lines) for lines in (CORPUS, VALID_CORPUS, TEST_CORPUS)]
        prompts = PROMPTS
    if args.max_tokens is not None:
        splits = [tokens[:args.max_tokens] for tokens in splits]
    if any(len(tokens) < max(N_VALUES) for tokens in splits):
        parser.error("Each split must contain at least four tokens.")
    vocab = build_vocab(splits[0], vocab_size)
    word_to_id = {word: i for i, word in enumerate(vocab)}
    unk_id = word_to_id["<unk>"]
    train_ids, valid_ids, test_ids = [to_ids(tokens, word_to_id) for tokens in splits]
    output_file = create_output_file()

    def report(message: str) -> None:
        print(message, flush=True)
        with output_file.open("a", encoding="utf-8") as handle:
            print(message, file=handle)

    report(f"Dataset: {'WikiText-2' if args.wikitext else 'built-in demo with held-out validation/test sentences'}")
    report(f"Tokens: train {len(train_ids):,}, validation {len(valid_ids):,}, test {len(test_ids):,}")
    report(f"Vocabulary: {len(vocab):,}; test <unk> share: {test_ids.count(unk_id) / len(test_ids):.1%}")
    report(f"D={dimensions}, epochs={epochs}, batch size={batch_size}, learning rate={learning_rate}, seed={SEED}")
    results = []
    for n in N_VALUES:
        report(f"\n===== n = {n}: read {n - 1} previous words, predict the next one =====")
        train_contexts, train_targets = make_windows(train_ids, n)
        valid_contexts, valid_targets = make_windows(valid_ids, n)
        test_contexts, test_targets = make_windows(test_ids, n)
        start = time.perf_counter()
        counts = train_count_model(train_contexts, train_targets)
        train_accuracy, _ = count_accuracy(counts, train_contexts[:200_000], train_targets[:200_000])
        count_test_accuracy, found = count_accuracy(counts, test_contexts, test_targets)
        report(f"Count model: {len(counts[-1]):,} full contexts, built/evaluated in {time.perf_counter() - start:.2f}s")
        report(f"  train accuracy {train_accuracy:.2%} (first up to 200,000 windows), test accuracy {count_test_accuracy:.2%}")
        report(f"  full test context seen: {found:.1%}")
        report("Neural model:")
        start = time.perf_counter()
        model = train_neural_model(
            train_contexts, train_targets, valid_contexts, valid_targets, len(vocab), n,
            dimensions, epochs, batch_size, learning_rate, report,
        )
        neural_accuracy, neural_loss = neural_evaluate(model, test_contexts, test_targets)
        report(f"  trained/evaluated in {time.perf_counter() - start:.2f}s; "
               f"test accuracy {neural_accuracy:.2%}, test loss {neural_loss:.3f}")
        results.append((n, count_test_accuracy, neural_accuracy))
        predictors: dict[str, Callable[[Context], list[float]]] = {
            "count": lambda context: count_probabilities(counts, context, len(vocab), unk_id),
            "neural": lambda context: neural_probabilities(model, context, unk_id),
        }
        for prompt in prompts:
            prompt_ids = to_ids(prompt.split(), word_to_id)
            for name, predictor in predictors.items():
                for method, rng in (("greedy", None), ("sampled", random.Random(SEED))):
                    ids = generate(predictor, prompt_ids, n - 1, args.generate_words, rng)
                    report(f"\n[n={n}, {name}, {method}, prompt={prompt!r}]\n"
                           + detokenize([vocab[i] for i in ids]))
    report("\n===== Summary: test accuracy =====")
    for n, count_score, neural_score in results:
        report(f"n={n}: count {count_score:.2%}, neural {neural_score:.2%}")
    report(f"\nAll generated text and results saved to {output_file}")


if __name__ == "__main__":
    main()
