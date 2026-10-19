# 02 · Negative sampling

**Interactive walkthrough:** [https://pratikktiwari.github.io/text-nn-from-scratch/02-negative-sampling/](https://pratikktiwari.github.io/text-nn-from-scratch/02-negative-sampling/) (source: [docs/02-negative-sampling](../../docs/02-negative-sampling/index.html)) replays the real training run pair by pair, including every random "wrong" word, with every number worked out.

## What we build

The same skip-gram model as [chapter 01](../01-word2vec-skipgram/README.md), on the same sentences, with the same starting numbers. Only the question the model answers changes.

In chapter 01, for every (center, context) pair the model scored **all** V words, turned the scores into chances with a softmax, and nudged every word's output weights. Here, it answers a few yes/no questions instead:

- "Is *furry* a real neighbour of *cats*?" The answer should be **yes**.
- "Is *bananas* a real neighbour of *cats*?" "Is *trucks*?" "Is *sweet*?" These are K words picked at random, and the answer should be **no**.

Each question touches only one word's output weights. So one training step costs the same whether the vocabulary has 21 words or 50,000. This is **skip-gram with negative sampling** (SGNS), the version of Word2Vec people actually train.

## Why chapter 01 is too slow

For one pair, the full softmax does work proportional to V twice: once to compute V scores (and the sum below the softmax), and once to update all V columns of `W`.

| | full softmax (ch01) | negative sampling (ch02) |
|---|---|---|
| words scored per pair | V | 1 + K |
| output weights changed per pair | V | 1 + K |
| V = 21, K = 3 | 21 | 4 |
| V = 50,000, K = 5 | 50,000 | 6 |

With a real vocabulary, the softmax is about ten thousand times more work per pair, and almost all of it is spent pushing down words that were never going to be picked anyway.

## Intuition

The goal of chapter 01 was never really to predict neighbours. Predicting was just the excuse that forced words with the same neighbours to get similar numbers. Any task with that effect works.

Negative sampling keeps the effect with a much cheaper task:

- **Pull together** the center word and the word really next to it.
- **Push apart** the center word and a few random words.

After many pairs, *cats*, *dogs* and *rabbits* have all been pulled toward the same words (*furry*, *playful*, *gentle*) and pushed away from random ones. So they end up close together, just like in chapter 01.

The random words are called **negatives** (the real neighbour is the **positive**). They aren't checked against the sentence. Sometimes a "wrong" word is really a neighbour somewhere else (for *cats*, a sampled *playful* is a real neighbour too). With many random draws, this noise averages out.

## The pieces

| Name | Shape | Meaning |
|---|---|---|
| `V` | number | vocabulary size; 21 for the toy sentences |
| `D` | number | how many numbers describe each word; 3 here |
| `K` | number | how many negatives we draw for every real pair; 3 here |
| `E` | V × D | the **embeddings**, one row per word. This is what we keep at the end. |
| `W` | V × D | the **output weights**, one **row** per word. Same numbers as chapter 01's `W`, laid out the other way so `W[word]` is a single row. |
| `h` | D | the center word's row of `E` |
| `u` | D | the positive (context) word's row of `W` |
| `u_neg` | D | one negative word's row of `W` |
| `counts` | V | how many times each word appears in the sentences |
| `negative_probabilities` | V | the chance of each word being drawn as a negative |

## Math

### 1. How negatives are drawn

Each word's chance of being drawn is its count raised to the power 0.75, divided by the total:

$$P(w) = \frac{\text{count}(w)^{0.75}}{\sum_{v} \text{count}(v)^{0.75}}$$

In our sentences, *pets*, *fruit* and *vehicles* appear 9 times each and every other word appears 3 times:

$$3^{0.75} = 2.2795 \qquad 9^{0.75} = 5.1962$$

$$\text{total} = 18 \times 2.2795 + 3 \times 5.1962 = 41.031 + 15.589 = 56.620$$

$$P(\text{cats}) = \frac{2.2795}{56.620} = 0.0403 \qquad P(\text{pets}) = \frac{5.1962}{56.620} = 0.0918$$

Without the power, *pets* would be drawn 3 times as often as *cats* (9 vs 3). With it, *pets* is drawn about 2.3 times as often. Common words are still drawn more, but rare words get a fairer share. The 0.75 is the value the Word2Vec authors found worked best.

If the drawn word is the real context word, we throw it away and draw again (`sample_negatives`). The same negative can be drawn twice.

### 2. The sigmoid

Each yes/no question needs a chance between 0 and 1 from a single score. The sigmoid gives that:

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

A large positive score gives a chance near 1 ("yes"), a large negative score gives a chance near 0 ("no"), and a score of 0 gives exactly 0.5. Two facts we use below:

$$1 - \sigma(x) = \frac{e^{-x}}{1 + e^{-x}} = \frac{1}{e^{x} + 1} = \sigma(-x)$$

and, by the chain rule on $(1 + e^{-x})^{-1}$,

$$\sigma'(x) = \frac{e^{-x}}{(1 + e^{-x})^2} = \frac{1}{1 + e^{-x}} \cdot \frac{e^{-x}}{1 + e^{-x}} = \sigma(x)\,\big(1 - \sigma(x)\big)$$

### 3. The positive pair

For center word $c$ and real neighbour $x$:

$$h = E_c \qquad u = W_x \qquad s = h \cdot u = \sum_{i=0}^{D-1} h_i\,u_i$$

The chance the model gives to "yes, $x$ is a neighbour" is $\sigma(s)$. The loss is the cross-entropy for that answer:

$$L_+ = -\ln \sigma(s) = -\ln \frac{1}{1 + e^{-s}} = \ln(1 + e^{-s})$$

The code computes $\ln(1 + e^{-s})$ as `np.logaddexp(0.0, -score)`, which gives the same value without overflowing for large $|s|$.

**Error at the score.** With $\frac{d}{du}(-\ln u) = -\frac{1}{u}$ and the sigmoid derivative from step 2:

$$\frac{dL_+}{ds} = -\frac{1}{\sigma(s)} \cdot \sigma(s)\,\big(1 - \sigma(s)\big) = -\big(1 - \sigma(s)\big) = \sigma(s) - 1$$

This is `error = sigmoid(score) - 1.0`. It is always negative: the chance we gave minus the chance we wanted (1). Compare chapter 01's $p_j - y_j$: it is the same "got minus wanted" rule.

**Gradients.** Since $s = \sum_i h_i u_i$, we have $\partial s / \partial h_i = u_i$ and $\partial s / \partial u_i = h_i$. So

$$\nabla_h L_+ = \big(\sigma(s) - 1\big)\,u \qquad \nabla_u L_+ = \big(\sigma(s) - 1\big)\,h$$

These are `gradient_h = error * u` and `gradient_u = error * h`. Both use $h$ and $u$ from before either one is updated.

**Update.**

$$E_c \leftarrow E_c - \eta\,\nabla_h L_+ \qquad W_x \leftarrow W_x - \eta\,\nabla_u L_+$$

Since the error is negative, subtracting it moves $E_c$ toward $u$ and $W_x$ toward $h$. The two vectors line up more, so next time their score is higher.

### 4. Each negative

For a drawn word $n$, we re-read $h$ (the positive update already moved $E_c$):

$$h = E_c \qquad u_n = W_n \qquad s_n = h \cdot u_n$$

Now the right answer is "no", whose chance is $1 - \sigma(s_n) = \sigma(-s_n)$:

$$L_n = -\ln \sigma(-s_n) = \ln(1 + e^{s_n})$$

In code: `np.logaddexp(0.0, score_neg)`.

**Error at the score.** Write $L_n = -\ln\big(1 - \sigma(s_n)\big)$. The derivative of $1 - \sigma(s_n)$ is $-\sigma(s_n)\big(1 - \sigma(s_n)\big)$, so

$$\frac{dL_n}{ds_n} = -\frac{1}{1 - \sigma(s_n)} \cdot \Big(-\sigma(s_n)\,\big(1 - \sigma(s_n)\big)\Big) = \sigma(s_n)$$

This is `error_neg = sigmoid(score_neg)`. Again "got minus wanted": the chance of "yes" we gave, minus the 0 we wanted. It is always positive.

**Gradients and update.**

$$\nabla_h L_n = \sigma(s_n)\,u_n \qquad \nabla_{u_n} L_n = \sigma(s_n)\,h$$

$$E_c \leftarrow E_c - \eta\,\sigma(s_n)\,u_n \qquad W_n \leftarrow W_n - \eta\,\sigma(s_n)\,h$$

Since the error is positive, $E_c$ moves away from $u_n$ and $W_n$ moves away from $h$. Their score goes down.

### 5. The whole loss for one pair

Adding the positive and the K negatives $n_1, \dots, n_K$:

$$L = -\ln \sigma(h \cdot u_x) \;-\; \sum_{k=1}^{K} \ln \sigma(-h \cdot u_{n_k})$$

At the start every score is close to 0, so every term is close to $\ln 2 = 0.6931$. With K = 3 that makes about $4 \ln 2 = 2.77$ per pair, which is exactly the first epoch's average loss (2.7682).

This is not the same number as chapter 01's loss ($\ln 21 \approx 3.04$ at the start), so the two curves can't be compared directly. Only the shape of each curve, and the embeddings they produce, can.

## Code walkthrough

The code is in [`skipgram-negative-sampling.py`](skipgram-negative-sampling.py). Everything up to the pairs is the same as chapter 01. The changes:

1. **Hyperparameters.** `EPOCHS = 50` (5 is too few for this task) and the new `K = 3`.
2. **`W` layout.** `W` stays V × D (not transposed), so `W[word]` is that word's row. The starting numbers are the same as chapter 01's, since `np.random.seed(42)` draws `E` and then `W` in the same order.
3. **Negative-sampling table.** `counts`, then `counts ** 0.75` normalised into `negative_probabilities` (math step 1).
4. **`sample_negatives(context, K)`.** Draws with `np.random.choice(vocab_length, p=negative_probabilities)` until it has K words that aren't the context word.
5. **`sigmoid(x)`.** Clips `x` to [−15, 15] before `np.exp`, so it never overflows.
6. **Training loop.** For each epoch it shuffles the pairs. Then for each pair it does the positive update (math step 3), draws K negatives, and does one update per negative (math step 4), re-reading `h` each time.
7. **Results.** The average loss per real pair goes into `losses`. Then, as in chapter 01, it prints the embeddings, plots the loss and the 2-D PCA view, and lists the most similar words.

## Run it

From the repository root:

```bash
source .venv/bin/activate
cd chapters/02-negative-sampling
python skipgram-negative-sampling.py
```

Each run writes a log to a new `skipgram-negative-sampling.output_N.txt` next to the script. These logs are gitignored.

## What to notice

- **The loss.** It starts at 2.77 ($4 \ln 2$) and barely moves for the first 5 epochs (2.7476 at epoch 5). Then it falls: 2.47 at epoch 10, 1.91 at 20, 1.49 at 30 and 1.21 at 50. In the early epochs every vector is tiny, so every score is close to 0 and the pushes and pulls nearly cancel. Once the vectors grow, the groups separate quickly.
- **The groups.** After 50 epochs, the three groups have clearly separated. The words most similar to *cats* are *pets* (0.998), *dogs* (0.997) and *rabbits* (0.997). For *dogs* they are *rabbits* (0.99999), *pets* and *cats*.
- **Why *pets* sits next to *cats*.** They share exactly the same neighbours: *furry*, *playful* and *gentle*. The model can't tell them apart, so they get the same embedding. *fruit* and *vehicles* behave the same way with their groups.
- **The describing words form their own groups.** *furry*, *playful* and *gentle* end up together, as do *sweet*, *juicy* and *fresh*, and *fast*, *motorized* and *wheeled*. Each set shares neighbours too.
- **Randomness.** Every negative is a random draw, so this run's numbers depend on the seed and on the exact order of draws. The walkthrough page replays NumPy's random generator exactly, so its numbers match the log line for line.

## Exercises

1. Set `K = 1`, then `K = 10`. How does the loss curve change, and do the groups still separate after 50 epochs?
2. Replace `counts ** 0.75` with `counts ** 1.0`, then with `counts ** 0.0` (every word equally likely). Count how often *pets* gets drawn in each case.
3. Remove the `if word != context` check in `sample_negatives`. What does the model now get told, and does training still work?
4. Run with `EPOCHS = 5` and look at the words most similar to *dogs*. Why does this task need more epochs than chapter 01?
5. Compare the final embeddings with chapter 01's (same seed, same starting numbers). Do the same groups form?

## What's next

**03 · Bigram language model.** So far, the models only learned *which words appear near each other*. They can't produce text. Chapter 03 builds the simplest model that can: given one word, it predicts the next one, and then we can sample from it to generate sentences. It brings back the softmax from chapter 01, now as the actual goal rather than an excuse.
