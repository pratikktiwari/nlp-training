"""Chapter 04: an n-gram language model from scratch with NumPy.

Both models read the previous n-1 words and predict the next word:
1. Count model: remember which word came next most often after each context.
2. Neural model: look up an embedding for each context word, join them, score every
   vocabulary word with one linear layer, and turn the scores into probabilities with softmax.
   Gradients are written by hand, like chapters 01 and 02.

Run:  python chapters/04-ngram-language-model/ngram-language-model.py
"""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import time
import urllib.request

import numpy as np


CHAPTER_DIR = Path(__file__).resolve().parent
DATA_DIR = CHAPTER_DIR / "data"
# WikiText-2: about 2 million words of Wikipedia articles, already split into words.
# The PyTorch examples repository keeps a copy that we download once into data/.
DATA_URL = "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/{split}.txt"

VOCAB_SIZE = 10_000      # the 9,999 most common training words, plus <unk> for every other word
N_VALUES = (2, 3, 4)     # n = 2 reads 1 previous word, n = 3 reads 2, n = 4 reads 3
D = 64                   # numbers per word embedding
LEARNING_RATE = 4.0     # large because each step averages the gradient over a whole batch
BATCH_SIZE = 512
EPOCHS = 1              # one pass over 2 million windows; more epochs help but take minutes each
GENERATE_WORDS = 60
PROMPTS = ["the game was", "in the first season of the", "the city is located"]
SEED = 42


# ---------- 1. Load the text ----------

def download_wikitext() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    for split in ("train", "valid", "test"):
        path = DATA_DIR / f"wiki.{split}.txt"
        if not path.exists():
            print(f"Downloading {split} split to {path} ...")
            urllib.request.urlretrieve(DATA_URL.format(split=split), path)


def read_tokens(path: Path) -> list[str]:
    """One long list of lowercase words. Each paragraph ends with <eos> ("end of sentence")."""
    tokens = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        # skip blank lines and article headings such as "= Valkyria Chronicles III ="
        if not line or line.startswith("="):
            continue
        tokens.extend(line.lower().split())
        tokens.append("<eos>")
    return tokens


def build_vocab(train_tokens: list[str], size: int) -> list[str]:
    counts = Counter(train_tokens)
    counts.pop("<unk>", None)  # WikiText already uses <unk> for rare words; we keep it as the last ID
    return [word for word, _ in counts.most_common(size - 1)] + ["<unk>"]


def to_ids(tokens: list[str], word_to_id: dict[str, int]) -> np.ndarray:
    unk = word_to_id["<unk>"]
    return np.array([word_to_id.get(word, unk) for word in tokens], dtype=np.int64)


def make_windows(ids: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Slide a window of n words along the text.

    The first n-1 words of each window are the context; the last word is the target.
    For "the cat sat on" and n = 3: (the, cat) -> sat, (cat, sat) -> on.
    """
    windows = np.lib.stride_tricks.sliding_window_view(ids, n)
    return windows[:, :-1], windows[:, -1]


# ---------- 2. Count model ----------

def train_count_model(contexts: np.ndarray, targets: np.ndarray) -> list[dict[tuple, int]]:
    """For every context length k = n-1, ..., 1, 0, remember the most common next word.

    best[k][last k context words] = the word seen most often right after them.
    Shorter contexts are the fallback for contexts that never appear in training.
    """
    context_length = contexts.shape[1]
    best = []
    for k in range(context_length + 1):
        # count every (context, next word) combination; k = 0 keeps no context, which is plain word frequency
        shortened = map(tuple, contexts[:, context_length - k:].tolist())
        ngram_counts = Counter(zip(shortened, targets.tolist()))
        best_count: dict[tuple, int] = {}
        best_word: dict[tuple, int] = {}
        for (context, word), count in ngram_counts.items():
            if count > best_count.get(context, 0):
                best_count[context] = count
                best_word[context] = word
        best.append(best_word)
    return best


def count_predict(best: list[dict[tuple, int]], context: list[int]) -> tuple[int, int]:
    """Return (predicted word ID, number of context words actually used)."""
    for k in range(len(context), -1, -1):
        key = tuple(context[len(context) - k:])
        if key in best[k]:
            return best[k][key], k
    raise RuntimeError("The empty context is always in the table.")


def count_accuracy(best: list[dict[tuple, int]], contexts: np.ndarray, targets: np.ndarray) -> tuple[float, float]:
    """Return (accuracy, share of windows whose full context was seen in training)."""
    correct = 0
    full_context_seen = 0
    for context, target in zip(contexts.tolist(), targets.tolist()):
        prediction, used = count_predict(best, context)
        correct += prediction == target
        full_context_seen += used == len(context)
    return correct / len(targets), full_context_seen / len(targets)


def count_probabilities(train_contexts: np.ndarray, train_targets: np.ndarray, context: list[int],
                        vocab_size: int, unk_id: int) -> np.ndarray:
    """Find every place in the training text where `context` appears and count the word after it.

    If the context never appears (or only <unk> follows it), drop its first word and search again.
    """
    context_length = train_contexts.shape[1]
    for k in range(len(context), -1, -1):
        matches = np.all(train_contexts[:, context_length - k:] == context[len(context) - k:], axis=1)
        counts = np.bincount(train_targets[matches], minlength=vocab_size).astype(np.float64)
        counts[unk_id] = 0
        if counts.sum() > 0:
            return counts / counts.sum()
    raise RuntimeError("The training text has no words besides <unk>.")


# ---------- 3. Neural model ----------

def init_neural_model(vocab_size: int, n: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    context_length = n - 1
    return {
        # E: one row of D numbers per word (the same idea as chapter 01's E)
        "E": (rng.standard_normal((vocab_size, D)) * 0.1).astype(np.float32),
        # W: one column per vocabulary word; it reads all n-1 embeddings joined end to end
        "W": (rng.standard_normal((context_length * D, vocab_size)) * 0.1).astype(np.float32),
        # b: one starting score per vocabulary word, so common words can score high in any context
        "b": np.zeros(vocab_size, dtype=np.float32),
    }


def forward(model: dict[str, np.ndarray], contexts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    batch_size, context_length = contexts.shape
    # look up every context word's embedding, then join them: (B, n-1, D) -> (B, (n-1)*D)
    x = model["E"][contexts].reshape(batch_size, context_length * D)
    scores = x @ model["W"] + model["b"]
    return x, scores


def softmax(scores: np.ndarray) -> np.ndarray:
    # subtracting each row's largest score changes nothing mathematically but keeps np.exp from overflowing
    shifted = scores - scores.max(axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / exp_scores.sum(axis=1, keepdims=True)


def train_step(model: dict[str, np.ndarray], contexts: np.ndarray, targets: np.ndarray, learning_rate: float) -> float:
    batch_size, context_length = contexts.shape
    rows = np.arange(batch_size)

    # forward pass
    x, scores = forward(model, contexts)
    probabilities = softmax(scores)
    loss = -np.log(probabilities[rows, targets] + 1e-12).mean()

    # backward pass: d(loss)/d(scores) = probabilities - one_hot(target), averaged over the batch
    d_scores = probabilities
    d_scores[rows, targets] -= 1.0
    d_scores /= batch_size
    d_W = x.T @ d_scores
    d_b = d_scores.sum(axis=0)
    d_x = d_scores @ model["W"].T  # uses W from before the update

    # update: subtract learning rate x gradient
    model["W"] -= learning_rate * d_W
    model["b"] -= learning_rate * d_b
    # split d_x back into one D-sized piece per context word and update those rows of E.
    # np.add.at adds correctly when the same word appears twice in a batch.
    np.add.at(model["E"], contexts, -learning_rate * d_x.reshape(batch_size, context_length, D))
    return float(loss)


def neural_evaluate(model: dict[str, np.ndarray], contexts: np.ndarray, targets: np.ndarray) -> tuple[float, float]:
    """Return (accuracy, average loss) without changing the model."""
    correct, total_loss = 0, 0.0
    for start in range(0, len(targets), 4096):
        batch_contexts = contexts[start:start + 4096]
        batch_targets = targets[start:start + 4096]
        _, scores = forward(model, batch_contexts)
        probabilities = softmax(scores)
        correct += int((scores.argmax(axis=1) == batch_targets).sum())
        total_loss += float(-np.log(probabilities[np.arange(len(batch_targets)), batch_targets] + 1e-12).sum())
    return correct / len(targets), total_loss / len(targets)


def train_neural_model(train_contexts, train_targets, valid_contexts, valid_targets, vocab_size, n, report):
    rng = np.random.default_rng(SEED)
    model = init_neural_model(vocab_size, n, rng)
    losses = []
    for epoch in range(EPOCHS):
        order = rng.permutation(len(train_targets))
        total_loss = 0.0
        for start in range(0, len(order), BATCH_SIZE):
            batch = order[start:start + BATCH_SIZE]
            total_loss += train_step(model, train_contexts[batch], train_targets[batch], LEARNING_RATE) * len(batch)
        losses.append(total_loss / len(order))
        valid_accuracy, valid_loss = neural_evaluate(model, valid_contexts, valid_targets)
        report(f"  epoch {epoch + 1}/{EPOCHS}: train loss {losses[-1]:.3f}, "
               f"validation loss {valid_loss:.3f}, validation accuracy {valid_accuracy:.2%}")
    return model, losses


# ---------- 4. Generate text ----------

def generate(next_word_probabilities, prompt_ids: list[int], context_length: int,
             words: int, rng: np.random.Generator | None) -> list[int]:
    """Repeat: read the last n-1 words, pick the next word, append it, slide the window.

    rng = None picks the most likely word every time (greedy).
    Otherwise draw the next word at random in proportion to its probability (sampling).
    """
    text = list(prompt_ids)
    for _ in range(words):
        probabilities = next_word_probabilities(text[-context_length:])
        if rng is None:
            text.append(int(probabilities.argmax()))
        else:
            text.append(int(rng.choice(len(probabilities), p=probabilities)))
    return text


def neural_probabilities(model: dict[str, np.ndarray], context: list[int], unk_id: int) -> np.ndarray:
    _, scores = forward(model, np.array([context]))
    probabilities = softmax(scores.astype(np.float64))[0]
    # never write <unk>: it stands for "some rare word", not a real word
    probabilities[unk_id] = 0
    return probabilities / probabilities.sum()


def detokenize(words: list[str]) -> str:
    """Join words back into readable text: <eos> becomes a paragraph break, and WikiText's
    split-up marks such as "1 @,@ 000" and "well @-@ known" are glued back together."""
    text = (" ".join(words) + " ").replace(" <eos> ", "\n\n").replace(" @-@ ", "-").replace(" @,@ ", ",").replace(" @.@ ", ".").strip()
    for mark in (" ,", " .", " ;", " :", " 's", " )", " n't"):
        text = text.replace(mark, mark.strip() if mark[1] in ",.;:)" else mark[1:])
    return text.replace("( ", "(")


def write_generated_text(samples: list[dict], vocab_size: int) -> None:
    """Save every generated paragraph to generated-text.md, grouped by n and prompt."""
    lines = ["# Generated text", "",
             f"Written by `ngram-language-model.py`: WikiText-2, vocabulary {vocab_size:,}, D = {D}, "
             f"{EPOCHS} epoch{'s' if EPOCHS > 1 else ''}, seed {SEED}.",
             f"Each text continues the prompt for {GENERATE_WORDS} words. Greedy always takes the most likely word; "
             "sampled draws a word at random by its probability. ¶ marks the end of a paragraph.", ""]
    for n in N_VALUES:
        lines += [f"## n = {n}: reads {n - 1} previous word{'s' if n > 2 else ''}", ""]
        for prompt in PROMPTS:
            lines += [f"### Prompt: \"{prompt}\"", ""]
            for sample in samples:
                if sample["n"] == n and sample["prompt"] == prompt:
                    lines += [f"**{sample['model']}, {sample['method']}:** " + sample["text"].replace("\n\n", " ¶ "), ""]
    (CHAPTER_DIR / "generated-text.md").write_text("\n".join(lines), encoding="utf-8")


# ---------- 5. Run everything ----------

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
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--export-web", action="store_true", help="also write results.json for the chapter's web page")
    args = parser.parse_args()
    output_file = create_output_file()

    def report(*values: object) -> None:
        print(*values)
        with output_file.open("a", encoding="utf-8") as handle:
            print(*values, file=handle)

    download_wikitext()
    train_tokens = read_tokens(DATA_DIR / "wiki.train.txt")
    valid_tokens = read_tokens(DATA_DIR / "wiki.valid.txt")
    test_tokens = read_tokens(DATA_DIR / "wiki.test.txt")
    vocab = build_vocab(train_tokens, VOCAB_SIZE)
    word_to_id = {word: i for i, word in enumerate(vocab)}
    unk_id = word_to_id["<unk>"]
    train_ids, valid_ids, test_ids = (to_ids(t, word_to_id) for t in (train_tokens, valid_tokens, test_tokens))

    report(f"Words: train {len(train_ids):,}, validation {len(valid_ids):,}, test {len(test_ids):,}")
    report(f"Vocabulary: {len(vocab):,} words; {np.mean(test_ids == unk_id):.1%} of test words are <unk>")
    report("First 30 training words:", " ".join(train_tokens[:30]))

    results = []
    samples = []
    for n in N_VALUES:
        report(f"\n===== n = {n}: read {n - 1} previous word{'s' if n > 2 else ''}, predict the next one =====")
        train_contexts, train_targets = make_windows(train_ids, n)
        valid_contexts, valid_targets = make_windows(valid_ids, n)
        test_contexts, test_targets = make_windows(test_ids, n)

        start = time.perf_counter()
        best = train_count_model(train_contexts, train_targets)
        count_seconds = time.perf_counter() - start
        count_train_accuracy, _ = count_accuracy(best, train_contexts[:200_000], train_targets[:200_000])
        count_test_accuracy, found = count_accuracy(best, test_contexts, test_targets)
        report(f"Count model: built in {count_seconds:.1f} s; {len(best[-1]):,} different {n - 1}-word contexts")
        report(f"  train accuracy {count_train_accuracy:.2%} (first 200,000 windows), test accuracy {count_test_accuracy:.2%}")
        report(f"  the full {n - 1}-word test context was seen in training {found:.1%} of the time; otherwise we used a shorter one")

        report("Neural model:")
        start = time.perf_counter()
        model, losses = train_neural_model(train_contexts, train_targets, valid_contexts, valid_targets, len(vocab), n, report)
        neural_seconds = time.perf_counter() - start
        neural_test_accuracy, neural_test_loss = neural_evaluate(model, test_contexts, test_targets)
        report(f"  trained in {neural_seconds:.1f} s; test accuracy {neural_test_accuracy:.2%}, test loss {neural_test_loss:.3f}")

        results.append({
            "n": n, "contexts": len(best[-1]), "countSeconds": round(count_seconds, 2), "neuralSeconds": round(neural_seconds, 2),
            "countTrainAccuracy": count_train_accuracy, "countTestAccuracy": count_test_accuracy, "fullContextSeen": found,
            "neuralTestAccuracy": neural_test_accuracy, "neuralTestLoss": neural_test_loss, "neuralTrainLosses": losses,
        })

        predictors = {
            "count": lambda context: count_probabilities(train_contexts, train_targets, context, len(vocab), unk_id),
            "neural": lambda context: neural_probabilities(model, context, unk_id),
        }
        for prompt in PROMPTS:
            prompt_ids = [word_to_id.get(word, unk_id) for word in prompt.split()]
            for model_name, predictor in predictors.items():
                for method, rng in (("greedy", None), ("sampled", np.random.default_rng(SEED))):
                    ids = generate(predictor, prompt_ids, n - 1, GENERATE_WORDS, rng)
                    text = detokenize([vocab[i] for i in ids])
                    samples.append({"n": n, "model": model_name, "method": method, "prompt": prompt, "text": text})
                    if prompt == PROMPTS[0]:
                        report(f"\n[{model_name}, {method}] {text}")

    report("\n===== Summary: test accuracy =====")
    report(f"{'n':>3} {'count model':>12} {'neural model':>13}")
    for row in results:
        report(f"{row['n']:>3} {row['countTestAccuracy']:>12.2%} {row['neuralTestAccuracy']:>13.2%}")

    write_generated_text(samples, len(vocab))
    report("\nAll generated paragraphs saved to", CHAPTER_DIR / "generated-text.md")

    if args.export_web:
        destination = CHAPTER_DIR.parents[1] / "docs" / "04-ngram-language-model" / "results.json"
        destination.write_text(json.dumps({
            "settings": {"vocabSize": len(vocab), "D": D, "epochs": EPOCHS, "batchSize": BATCH_SIZE,
                         "learningRate": LEARNING_RATE, "seed": SEED, "generateWords": GENERATE_WORDS,
                         "trainWords": int(len(train_ids)), "testWords": int(len(test_ids))},
            "results": results, "samples": samples,
        }, indent=1) + "\n", encoding="utf-8")
        report("Web results written to", destination)
    report("Output saved to", output_file)


if __name__ == "__main__":
    main()
