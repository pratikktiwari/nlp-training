# 01 · Word2Vec: skip-gram

**The idea:** learn a vector for each word by asking it to predict nearby words. Words that share neighbours receive similar training signals, which can make their vectors similar.

You only need basic Python, arrays, and the idea of taking a small step to reduce an error. The gradients are explained below; you do not need to derive them before running the example.

**Use this chapter in order:** understand the training pairs, follow one prediction and update, then run the script. The longer calculus derivation is optional.

- [Code](skipgram-word2vec.py)
- [Word2Vec intuition: predictions, loss, and learning](https://pratikktiwari.github.io/text-nn-from-scratch/01-word2vec-skipgram/intuition.html) ([page source](../../docs/01-word2vec-skipgram/intuition.html))
- [Interactive walkthrough](https://pratikktiwari.github.io/text-nn-from-scratch/01-word2vec-skipgram/) ([page source](../../docs/01-word2vec-skipgram/index.html))
- [Next chapter: negative sampling](../02-negative-sampling/README.md)

## What we build

An **embedding** is a short list of learned numbers representing a word. Here each word gets three numbers, such as `[0.2, -0.1, 0.4]`. This is an illustration, not a trained result.

The script learns embeddings from 27 short sentences about pets, fruit, and vehicles. There are 21 distinct words. Everything is NumPy: we calculate predictions, errors, and weight updates ourselves.

At the end, we keep the embedding table and ask which words have similar vectors. We are **not** building a text generator, and individual coordinates do not have assigned meanings such as "animal" or "speed".

**Why learn vectors at all?** Word IDs only identify words: ID 8 is not twice as meaningful as ID 4. Learned coordinates let us compare patterns of usage and supply numerical inputs to later models. Context prediction is the teaching task that gives those coordinates feedback, not the final reason we want them.

## Intuition

### 1. Turn a sentence into training examples

For the sentence `cats furry pets`, a window of 1 means "look one word left and one word right". Do not cross sentence boundaries.

| Center word (input) | Context word (target) |
|---|---|
| cats | furry |
| furry | cats |
| furry | pets |
| pets | furry |

Each row is a separate training example. **Skip-gram predicts a context word from one center word**, not the other way around. A context can be on either side, so this is not just next-word prediction.

The full corpus produces 108 pairs: 27 sentences, each producing the four pairs above.

### 2. Make a prediction

For `(cats, furry)`, give the model only `cats`. It assigns a probability to **every** word in the vocabulary. The target is `furry`.

When it sees `(cats, playful)` later, the input is still `cats`, but the target changes. There is no contradiction: the model learns a distribution over possible neighbours, not one permanent answer.

### 3. Why does this teach word similarity?

The corpus contains `cats furry pets`, `dogs furry pets`, and `rabbits furry pets`, plus matching sentences with `playful` and `gentle`.

`cats`, `dogs`, and `rabbits` must all learn to predict the same neighbours. Their embeddings receive related updates, encouraging similar representations. This is a consequence of shared contexts, not a rule telling the model that these words are animals.

Five short training epochs are not enough to guarantee neat groups. We will separate what the method aims to learn from what this small default run actually shows.

## Math

**Notation:** `log` means the natural logarithm, $\log_e$, with base $e \approx 2.718$. This is what `np.log` computes. The interactive walkthrough includes synchronized curves and tangent lines for softmax, loss, score error, and individual weight gradients.

### The two tables

The model has two sets of adjustable numbers, called **weights**:

| Name in code | Shape | Job |
|---|---|---|
| `E` | `V x D` = `21 x 3` | One input embedding per word, stored as a row |
| `W` | `D x V` = `3 x 21` | One output vector per candidate context word, stored as a column |
| `h = E[center].copy()` | `D` = `3` | The current center word's embedding |
| `scores`, `probabilities`, `error` | `V` = `21` | One value per candidate context word |

`V` is the vocabulary size; the script calls it `vocab_length`. `D` is the embedding size. A word ID is just a row or column index, not an embedding.

**Why two tables?** A word has two roles: input word and candidate neighbour. `E` represents the first role; `W` represents the second. Training uses both, but this script uses only `E` for the final similarity comparisons.

### One training step: predict, measure, update

Let `c` be the center word's ID and `x` the true context word's ID. The subscript `j` means any candidate word.

**1. Look up the center embedding.**

$$h = E_c$$

This selects a row. There is no need to multiply a one-hot vector by the entire table.

**2. Give every candidate a score.**

$$s_j = h \cdot W_{:,j}
\qquad\text{or, for all words at once,}\qquad s = hW$$

A dot product multiplies matching coordinates and adds the results. A higher score means a better fit according to the current weights. Scores can be negative and are not probabilities yet.

**3. Convert the scores to probabilities with softmax.**

We need candidates to compete for a fixed probability budget before judging the prediction. Exponentiation makes every weight positive; dividing by the total makes them sum to 1. Without that division, raising every score could raise every output instead of expressing a preference.

$$p_j = \frac{e^{s_j}}{\sum_k e^{s_k}}$$

All probabilities are positive and sum to 1. Because they share a denominator, the candidates compete for probability.

The code subtracts the largest score before exponentiating. This leaves the probabilities unchanged while avoiding very large exponentials.

**4. Measure the error with cross-entropy loss.**

Probabilities describe the prediction but do not yet give us an objective to minimize. The negative logarithm converts the observed target's probability into a cost, with a large penalty for confidently missing it.

$$L = -\log_e p_x$$

The negative sign turns log probabilities into costs. A high probability for the true context gives a small loss; a low probability gives a large loss.

For example, increasing `p_x` from `0.1` to `0.5` reduces the loss from about `2.303` to `0.693`. The code adds `1e-12` inside the logarithm to avoid taking `log(0)`.

**5. Work out which scores should change.**

Let `y` be a target vector: 1 at the true context's index, 0 everywhere else.

$$\delta = p - y$$

This is the gradient of the ideal softmax cross-entropy loss with respect to the scores. A **gradient** tells us how the loss changes when a weight changes; subtracting a small multiple of it moves downhill.

In the code, `delta` is named `error`:

```python
error = probabilities.copy()
error[context] -= 1
```

For the true context, `p_x - 1` is negative, asking for a higher score. For every other word, `p_j` is positive, asking for a lower score.

**6. Pass that error back to the tables and update them.**

$$\nabla_W L = h \otimes \delta
\qquad
\nabla_h L = W\delta$$

The symbol `nabla` ($\nabla$) means gradient. The outer product ($\otimes$) multiplies each entry of `h` by each entry of `delta`, making a `D x V` table.

```python
gradient_W = np.outer(h, error)
gradient_h = W @ error

W -= LEARNING_RATE * gradient_W
E[center] -= LEARNING_RATE * gradient_h
```

Both gradients must use the **old** weights, before either update. The learning rate controls the step size.

Only the center row of `E` changes. All columns of `W` participate, because every vocabulary word received a score.

### A small numerical example

To make the arithmetic visible, imagine a vocabulary of only three words. These are illustrative scores, not values from the seeded run.

For the pair `(cats, furry)`:

| Candidate | Score | `exp(score)` | Probability `p` | Target `y` | Error `p - y` |
|---|---|---|---|---|---|
| cats | `0` | `1` | `0.25` | `0` | `+0.25` |
| furry | `log(2)` | `2` | `0.50` | `1` | `-0.50` |
| pets | `0` | `1` | `0.25` | `0` | `+0.25` |

The exponential values sum to 4, so `furry` gets probability `2/4`. The loss is `-log(0.5)`, about `0.693`.

Subtracting the gradients nudges the output vector for `furry` in the direction of the center embedding, and the other output vectors in the opposite direction. The center embedding changes too, using contributions from all three output vectors.

<details>
<summary>Optional: where do the gradients come from?</summary>

Write $p_x = e^{s_x}/Z$, where $Z = \sum_k e^{s_k}$.

Differentiating the softmax gives two cases:

$$\frac{\partial p_x}{\partial s_j} =
\begin{cases}
p_x(1-p_x), & j=x \\
-p_xp_j, & j\ne x
\end{cases}$$

Since $\partial L/\partial p_x = -1/p_x$, the chain rule gives:

$$\frac{\partial L}{\partial s_j} =
\begin{cases}
p_x-1, & j=x \\
p_j, & j\ne x
\end{cases}
= p_j-y_j$$

Now use $s_j = \sum_i h_iW_{ij}$. Its derivatives are
$\partial s_j/\partial W_{ij}=h_i$ and $\partial s_j/\partial h_i=W_{ij}$:

$$\frac{\partial L}{\partial W_{ij}}=h_i\delta_j
\qquad
\frac{\partial L}{\partial h_i}=\sum_j W_{ij}\delta_j$$

These are exactly `np.outer(h, error)` and `W @ error`.

</details>

## Code walkthrough

Read [`skipgram-word2vec.py`](skipgram-word2vec.py) in these blocks:

| Block | What to look for |
|---|---|
| Prepare text | `corpus`, `sentences`, sorted `vocab`, and word-to-ID mappings |
| Create examples | `pairs`: each center and each neighbour within `WINDOW` |
| Initialize weights | Seed 42; small random values for `E` and `W`; transpose `W` so candidates are columns |
| Train | Shuffle pairs, then do the six steps above for each pair |
| Inspect results | Print `E`, draw loss and embedding plots, list similar words |

Defaults: `WINDOW = 1`, `D = 3`, `LEARNING_RATE = 0.05`, `EPOCHS = 5`.

The small random initialization matters: if both `E` and `W` started at zero, `h` would be zero, `np.outer(h, error)` would be zero, and `W @ error` would also be zero. Neither table could learn. The fixed seed makes the nonzero starting point reproducible.

An **epoch** is one pass through all 108 pairs. An **update** handles one pair. The reported loss is the average over the epoch, even though the console label says `Total Loss`.

## Run it

For a first run, create the environment and install dependencies from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python chapters/01-word2vec-skipgram/skipgram-word2vec.py
```

For later runs, activate the existing environment and run the final command again. The activation command above is for macOS/Linux; Windows PowerShell uses `.\.venv\Scripts\Activate.ps1`.

The script:

1. Prints the corpus, mappings, training pairs, initial weights, and epoch losses.
2. Prints each learned embedding.
3. Opens a loss plot and a 2-D embedding plot. **Close the plot windows to let the script continue to the similar-word output.**
4. Prints the three words most similar to `cats` and to `dogs`.

It also writes a new `skipgram-word2vec.output_N.txt` beside the script on each run. These logs are gitignored; no existing log is overwritten.

## What to notice

- **Early loss:** with nearly equal scores, each word gets probability about `1/21`. The loss is therefore near `log(21)`, about `3.04`. Five epochs only lower it slightly; this default is a short demonstration, not a well-trained embedding model.
- **Why loss does not need to reach zero:** the same input has several valid targets. A model cannot give probability 1 to all of them at once.
- **Similarity is a separate calculation:** training scores compare an input vector from `E` with an output vector from `W`. The final `cosine_similarity` compares two vectors from `E`, using their directions rather than just their lengths.
- **The plot is a projection:** the script normalizes the 3-D embeddings, then uses principal component analysis (PCA) to show them in 2-D. Some information is lost; use the cosine rankings as well as the picture.
- **Shared context is not identical meaning:** `cats` and `pets` have the same immediate neighbours in this corpus. The model can learn similar vectors for them even though one word is a broader category.

**Why keep `E` after training?** Its rows now carry information that helped solve many context predictions. To compare two of those rows, the script uses cosine similarity:

$$\operatorname{cosine}(a,b) = \frac{a\cdot b}{\lVert a\rVert\,\lVert b\rVert}$$

Dividing by both lengths isolates direction rather than magnitude. For an illustrative pair of unit vectors `[1, 0]` and `[0.8, 0.6]`, the dot product and cosine are `0.8`: fairly aligned. Values near 1 mean similar directions, 0 means perpendicular, and -1 means opposite. This is an inspection of the learned representations, not a probability or the training loss; the code adds `1e-12` to the denominator as a numerical guard.

Before moving on, explain the loop in one sentence: **use `E` and `W` to predict, use loss to judge, use gradients to assign responsibility, then subtract a small gradient step to store the lesson.**

## Exercises

Change one setting at a time in the script:

1. Raise `EPOCHS` from 5 to 50. Compare loss and the neighbours of `cats` before and after. Does a lower loss always mean every ranking looks sensible?
2. Set `WINDOW = 2`. Each three-word sentence now has six directed pairs instead of four. Confirm that the script produces 162 pairs, and identify the newly added ones.
3. Set `D = 2`. The embeddings now have two coordinates, but the plot still normalizes and applies PCA. Compare the similarity rankings, not just the axis positions.
4. Try `LEARNING_RATE = 0.3`, then `1.0`. Watch whether the loss decreases smoothly or becomes unstable; a larger step is not always better.

## What's next

[Chapter 02: negative sampling](../02-negative-sampling/README.md) keeps the same training pairs and the goal of learning embeddings, but changes the training task.

Here every pair scores all `V` words and updates all `V` output vectors. The next chapter replaces that vocabulary-wide competition with one positive example and a few sampled negative examples.
