# 02 · Negative sampling

**The idea:** instead of scoring every word for every training pair, score the observed neighbour and a few randomly sampled words.

Read [chapter 01](../01-word2vec-skipgram/README.md) first. You should know what a `(center, context)` pair is, why we have two weight tables, and how subtracting a gradient updates them. This chapter explains the new loss and sampling rule.

**Use this chapter in order:** compare the two training tasks, follow one yes/no update, then read how the script repeats it. The longer calculus derivation is optional.

- [Code](skipgram-negative-sampling.py)
- [Speed comparison with full softmax](speed-comparison.py)
- [Negative sampling intuition: data, noise, and learning](https://pratikktiwari.github.io/text-nn-from-scratch/02-negative-sampling/intuition.html) ([page source](../../docs/02-negative-sampling/intuition.html))
- [Interactive walkthrough](https://pratikktiwari.github.io/text-nn-from-scratch/02-negative-sampling/) ([page source](../../docs/02-negative-sampling/index.html))
- [Previous chapter: full-softmax skip-gram](../01-word2vec-skipgram/README.md)

## What we build

We learn word embeddings from the same 27 sentences, 21 words, and 108 training pairs as chapter 01. This version is **skip-gram with negative sampling**, or **SGNS**.

The important change is the training question:

| | Chapter 01: full softmax | Chapter 02: negative sampling |
|---|---|---|
| Question | Which vocabulary word is the context? | Does this pair come from the observed data rather than the noise samples? |
| Scores per real pair | All `V` words | One positive and `K` negatives |
| Output | Probabilities across words that sum to 1 | A separate sigmoid output for each scored pair |
| Training target | The real context's word ID | 1 for the positive pair, 0 for each negative |

**This is a different objective, not a cheaper way to compute the same softmax.** Its sigmoid outputs do not form a probability distribution over the vocabulary.

The reason for learning embeddings is unchanged: word IDs only identify words, while learned coordinates can capture shared usage and become inputs to later models. The observed-pair-versus-noise task supplies feedback without requiring a human to write down a "correct" vector.

## Intuition

### 1. Keep the real pair; add some noise

Take the real training pair `(cats, furry)`. For illustration, suppose the sampler draws `bananas`, `trucks`, and `sweet`:

| Center | Candidate context | Label | Why |
|---|---|---|---|
| cats | furry | 1 (positive) | Observed together in a training pair |
| cats | bananas | 0 (negative) | Drawn from the noise distribution |
| cats | trucks | 0 (negative) | Drawn from the noise distribution |
| cats | sweet | 0 (negative) | Drawn from the noise distribution |

We want a high score for the first pair and low scores for the other three. We do not score the rest of the vocabulary for this example.

**Negative does not mean "known never to be a neighbour".** It means "sampled as noise for this update". For `cats`, a sampled `playful` is also a real neighbour elsewhere in the corpus. The task learns a statistical distinction across many examples, not a perfect yes/no dictionary of neighbours.

### 2. Why can the embeddings still become useful?

`cats`, `dogs`, and `rabbits` all occur with `furry`, `playful`, and `gentle`. Training encourages their input embeddings to give high dot products with those shared output vectors.

The negatives discourage giving high scores to everything. Together, these signals can give words with shared contexts similar embeddings.

"Pull together" and "push apart" are useful shorthand for raising and lowering **dot products between input and output vectors**. They do not mean we directly move `E[cats]` toward `E[furry]`, or that we minimize Euclidean distance.

### 3. What work does this save?

| Vocabulary and sample size | Full-softmax scores per real pair | Negative-sampling scores per real pair |
|---|---|---|
| `V = 21`, `K = 3` | 21 | 4 |
| `V = 50,000`, `K = 5` | 50,000 | 6 |

The dot products and weight updates cost roughly `O(V * D)` for full softmax versus `O((K + 1) * D)` for negative sampling, where `D` is embedding size.

This compares **scoring and updating**, not measured runtime. The teaching script uses `np.random.choice` with a vocabulary-sized probability array and rejection sampling. Sampling has its own cost; the whole script is not guaranteed to take constant time as `V` grows. [Time it](#time-it) below measures this.

## Math

**Notation:** `log` means the natural logarithm, $\log_e$, with base $e \approx 2.718$, matching `np.log`. The interactive walkthrough plots the sigmoid's slope and compares positive and negative losses at the selected pair's score, so the signs of their errors are visible before the derivation.

### The tables and settings

| Name in code | Shape or default | Meaning |
|---|---|---|
| `vocab_length` (`V` below) | 21 | Number of distinct words |
| `D` | 3 | Coordinates per vector |
| `K` | 3 | Negative draws per real pair |
| `E` | `V x D` | Input embeddings; one row per center word |
| `W` | `V x D` | Output vectors; one row per candidate context word |
| `h` | `D` | Copy of the current center's row from `E` |
| `u` or `u_neg` | `D` | Copy of the candidate's row from `W` |

**Layout change:** chapter 01 stored each output vector as a column of `W`. Here it is a row, so we can use `W[word]` directly. With the same seed and initialization order, the starting word vectors are the same, just transposed in storage.

### 1. Draw the negative words

Count how often each word appears, raise each count to `0.75`, and normalize:

$$q(w) = \frac{\operatorname{count}(w)^{0.75}}
{\sum_v \operatorname{count}(v)^{0.75}}$$

This is `negative_probabilities` in the code. It is a **sampling distribution**, not a model prediction.

In this corpus, `pets`, `fruit`, and `vehicles` occur 9 times each; the other 18 words occur 3 times each.

| Word type | Raw count | Weight `count ** 0.75` | Base sampling probability |
|---|---|---|---|
| `cats` (or any count-3 word) | 3 | about 2.2795 | about 0.0403 |
| `pets` (or any count-9 word) | 9 | about 5.1962 | about 0.0918 |

The power flattens the frequencies: a count-9 word is about 2.28 times as likely as a count-3 word, rather than 3 times as likely.

Why not just use the raw counts? That would concentrate more noise examples on the most common words. Uniform sampling would remove frequency information entirely. The `0.75` power is a design choice between those extremes, not a probability that a word is a true neighbour.

`sample_negatives(context, K)` then applies these rules:

- Reject the current true context and draw again. Accepted draws therefore follow `q` conditioned on not being that context.
- Do not exclude other real neighbours or the center word.
- Allow repeats. Three draws do not necessarily mean three distinct words.

### 2. Score one pair and turn the score into a yes/no output

The vector lookup alone cannot answer the training question. The dot product combines the center and candidate coordinates into one adjustable compatibility score. Because that score is unbounded, we use sigmoid to put the prediction on a binary probability scale before comparing it with a label.

For a center embedding `h` and candidate output vector `u`:

$$s = h \cdot u
\qquad
p = \sigma(s) = \frac{1}{1+e^{-s}}$$

The **sigmoid** maps one score to a number between 0 and 1:

| Score `s` | Sigmoid output `p` |
|---|---|
| `-2` | about `0.119` |
| `0` | `0.5` |
| `2` | about `0.881` |

Unlike softmax, this does not use any other word's score. Think of `p` as the model's positive-label probability for this training task, **not** the probability that this word is the next word or a neighbour in general.

### 3. Measure the error

The sigmoid says what the model predicts, not how costly a mistake is. A loss supplies a differentiable objective: it must reward a high positive-label probability for the data example and a low one for noise. Without the negative terms, the task would not penalize indiscriminately high scores.

Let the label be `y`: 1 for the observed pair, 0 for a noise pair. Binary cross-entropy penalizes the wrong answer:

$$L = -y\log p - (1-y)\log(1-p)$$

The two cases are:

| Pair type | Label | Loss | Gradient at the score, `p - y` |
|---|---|---|---|
| Positive | 1 | $-\log\sigma(s)$ | $\sigma(s)-1$ |
| Negative | 0 | $-\log(1-\sigma(s))$ | $\sigma(s)$ |

A positive wants a larger score; a negative wants a smaller score. The same "prediction minus target" rule appears in chapter 01, but here it applies to one binary decision.

For numerical stability, the code calculates the positive loss as `np.logaddexp(0.0, -score)` and the negative loss as `np.logaddexp(0.0, score_neg)`. These equal `log(1 + exp(-s))` and `log(1 + exp(s))` without directly forming a potentially overflowing exponential.

### 4. Update the two vectors

With `error = p - y` and learning rate $\eta$:

$$\nabla_h L = \text{error}\,u
\qquad
\nabla_u L = \text{error}\,h$$

$$E_c \leftarrow E_c - \eta\,\nabla_h L
\qquad
W_{\text{candidate}} \leftarrow W_{\text{candidate}} - \eta\,\nabla_u L$$

Both gradients use copies of the vectors from **before** that binary update. Only the current center row of `E` and the candidate row of `W` change.

**Small worked example.** Use illustrative two-coordinate vectors, not the script's initialized values:

```text
h = [0.2, -0.1]
u = [0.4,  0.2]
label = 1
learning rate = 0.05
```

1. Score: `0.2 * 0.4 + (-0.1) * 0.2 = 0.06`.
2. Sigmoid output: about `0.5150`.
3. Positive loss: `-log(0.5150)`, about `0.6636`.
4. Error: `0.5150 - 1 = -0.4850`.
5. Gradients: `gradient_h = [-0.1940, -0.0970]` and `gradient_u = [-0.0970, 0.0485]`.
6. Updated vectors: `h = [0.2097, -0.09515]` and `u = [0.40485, 0.197575]`, approximately.

Their dot product has increased to about `0.0661`. The positive pair now gets a higher score.

For a negative pair, the error would instead be `+0.5150` at the original score. Subtracting that gradient takes the update in the opposite direction.

### 5. Repeat for the whole training pair

The script does **one positive update, followed by `K` negative updates**. It re-reads the center embedding before each negative because the previous update already changed it. A repeated negative is updated again using its latest vector.

At a fixed set of weights, the usual combined SGNS loss is:

$$L_{\text{pair}} =
-\log\sigma(h\cdot u_x)
-\sum_{k=1}^{K}\log\sigma(-h\cdot u_{n_k})$$

Here `x` is the true context and `n_k` is the kth sampled negative.

**Implementation detail:** this script updates immediately after each term rather than accumulating all gradients at one fixed `h`. Its logged loss sums terms evaluated just before their individual updates. Do not expect an implementation with one combined update to produce identical numbers.

Near initialization, all scores are close to 0, so each binary loss is about `log(2)`. With `K = 3`, the loss per real pair starts near `4 * log(2)`, about `2.77`.

<details>
<summary>Optional: where do the gradients come from?</summary>

The sigmoid derivative is:

$$\sigma'(s)=\sigma(s)(1-\sigma(s))$$

For a positive pair:

$$\frac{d[-\log\sigma(s)]}{ds}
=-\frac{\sigma(s)(1-\sigma(s))}{\sigma(s)}
=\sigma(s)-1$$

For a negative pair:

$$\frac{d[-\log(1-\sigma(s))]}{ds}
=\frac{\sigma(s)(1-\sigma(s))}{1-\sigma(s)}
=\sigma(s)$$

Both are `p - y`. Since $s=\sum_i h_i u_i$, its derivatives with respect to `h` and `u` are `u` and `h`. The chain rule gives the two vector gradients above.

The script clips sigmoid inputs to `[-15, 15]` to avoid overflow. The derivation describes the mathematical sigmoid; clipping approximates its extreme values in the implementation.

</details>

## Code walkthrough

Read [`skipgram-negative-sampling.py`](skipgram-negative-sampling.py) in these blocks:

| Block | What to look for |
|---|---|
| Prepare text and pairs | Same corpus, vocabulary, and window rule as chapter 01 |
| Initialize weights | Seed 42; `E` and `W` both have one row per word |
| Prepare sampling | `counts`, `negative_probabilities`, and `sample_negatives` |
| Positive update | Target 1; `error = sigmoid(score) - 1.0` |
| Negative updates | Draw `K` negatives; target 0; `error_neg = sigmoid(score_neg)` |
| Inspect results | Average loss per real pair, learned vectors, plots, and cosine rankings |

Defaults: `WINDOW = 1`, `D = 3`, `K = 3`, `LEARNING_RATE = 0.05`, `EPOCHS = 50`.

Small random starting vectors let both tables receive nonzero learning signals. Initializing both tables to zero would give a zero dot-product gradient with respect to either vector. Seed 42 makes the nonzero start repeatable; it does not remove the need to learn.

One epoch visits all 108 real pairs. Each real pair triggers four binary updates at the default `K`, giving 432 binary updates per epoch. Chapter 01 defaults to only 5 epochs, so the default runs are not a controlled comparison of convergence speed.

## Run it

If chapter 01's environment is already installed, run these commands from the repository root:

```bash
source .venv/bin/activate
python chapters/02-negative-sampling/skipgram-negative-sampling.py
```

For a first run, create the environment and install dependencies first:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python chapters/02-negative-sampling/skipgram-negative-sampling.py
```

The activation command above is for macOS/Linux; Windows PowerShell uses `.\.venv\Scripts\Activate.ps1`.

The script prints the data, initial weights, sampling probabilities, epoch losses, and learned embeddings. It opens loss and embedding plots. **Close the plot windows to let it continue to the similar-word output.**

Each run also writes a new `skipgram-negative-sampling.output_N.txt` beside the script. These logs are gitignored; no existing log is overwritten.

### Time it

The script now prints how long training took, measured with `time.perf_counter()` around the training loop. With 21 words that is well under a second. At this size there is nothing to save: scoring 21 words is cheap, and negative sampling does 4 small updates per pair instead of 1.

To see the difference, run the speed comparison. It makes up bigger corpora with the same sentence shape (`thing describing-word group`), and trains chapter 01's loop and this chapter's loop on the same 2,000 pairs:

```bash
python chapters/02-negative-sampling/speed-comparison.py
```

One run on a 4-core machine (seconds for the same 2,000 pairs; your numbers will differ, the pattern should not):

| Vocabulary | Full softmax (chapter 01) | Negative sampling, `np.random.choice` (chapter 02) | Negative sampling, table draw |
|---|---|---|---|
| 21 | 0.026 | 0.174 | 0.092 |
| 994 | 0.045 | 0.180 | 0.066 |
| 9,996 | 0.180 | 0.660 | 0.105 |
| 49,994 | 1.364 | 2.336 | 0.067 |

The vocabularies are not round numbers because each made-up group has exactly 7 words.

What this shows:

- **Full softmax slows down as the vocabulary grows.** Every pair computes `V` scores and changes all `V` columns of `W`: about 50 times slower at 50,000 words than at 21.
- **For small vocabularies full softmax wins.** At 21 words it is the fastest of the three, because one NumPy call over 21 numbers is cheaper than 4 separate small updates.
- **This chapter's sampler is slower still.** `np.random.choice(vocab_length, p=negative_probabilities)` re-reads all `V` probabilities on every single draw, and each call has a fixed start-up cost. So in this script, negative sampling is not faster than full softmax at any size.
- **Drawing from a table fixes it.** The original word2vec code fills a big table once, where each word takes a share of the slots equal to its sampling probability. A draw is then "pick a random slot", which costs the same for 21 or 50,000 words. With that, the training time stays flat as `V` grows: at 50,000 words it is about 20 times faster than full softmax. This is the speed-up negative sampling promises.

We keep `np.random.choice` in the main script because the interactive walkthrough replays its exact random draws.

## What to notice

- **Loss can move slowly at first.** Small initial vectors give scores near zero. Use the 50-epoch run to examine the learned structure rather than judging it after five epochs.
- **Do not compare raw loss with chapter 01.** Chapter 01 measures a vocabulary-wide classification loss; this chapter sums `K + 1` binary losses per real pair.
- **Changing `K` changes the loss scale.** More negatives add more terms. A larger loss after increasing `K` does not by itself mean worse embeddings.
- **Similar context, similar vectors:** look for `cats`, `dogs`, and `rabbits` to group together. `pets` can join them because it shares the same immediate neighbours. Similar contexts encourage similarity; they do not guarantee identical vectors.
- **Use rankings as well as the plot.** Cosine similarity compares rows of `E`. The plot normalizes those embeddings and uses PCA to reduce three dimensions to two, which can hide some differences.
- **Same initialization does not mean the same training path.** Sampling consumes random numbers, and the objectives differ. Changing `K` or sampling rules changes later draws and updates, even with seed 42.

**Why compare embeddings after training?** The input rows have been adjusted to solve many related decisions through shared output vectors. Comparing two rows of `E` by [cosine similarity](../01-word2vec-skipgram/README.md#what-to-notice) asks whether their learned directions are similar, rather than whether one candidate gets a positive label. That makes the representations reusable even though `W` and sigmoid were needed to teach them.

Before moving on, distinguish **prediction** (score and sigmoid), **judgment** (binary loss), **responsibility** (gradients), and **learning** (the immediate parameter update). None of the first three alone changes an embedding.

## Exercises

Change one thing at a time:

1. Try `K = 1`, then `K = 10`. Write down the near-initialization loss you expect: `(K + 1) * log(2)`. Compare the final cosine rankings instead of comparing raw losses across these runs.
2. Replace `counts ** 0.75` with `counts ** 1.0`, then `counts ** 0.0`. How does the printed base sampling probability for `pets` change? With exponent 0, each word should have probability `1/21`.
3. Find where `sample_negatives` rejects the positive context. If that check were removed, how could the same `(center, context)` pair receive both labels within one training step?
4. Set `EPOCHS = 5`, then restore 50. Compare the neighbours of `dogs`. What changes that the early loss alone did not reveal?
5. Run both chapters for 50 epochs with the same `D` and learning rate. Compare word groups and rankings, but not raw loss values or individual coordinates: their objectives and random update sequences differ.

## What's next

In [chapter 03: GloVe](../03-glove/README.md), aggregate context evidence into co-occurrence counts, then fit their logarithms with vectors and biases. It introduces PyTorch autograd and Adagrad, using a different corpus and a regression objective rather than data-versus-noise classification.
