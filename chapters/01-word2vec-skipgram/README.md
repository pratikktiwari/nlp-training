# 01 · Word2Vec: skip-gram

**Interactive walkthrough:** [https://pratikktiwari.github.io/text-nn-from-scratch/01-word2vec-skipgram/](https://pratikktiwari.github.io/text-nn-from-scratch/01-word2vec-skipgram/) (source: [docs/01-word2vec-skipgram](../../docs/01-word2vec-skipgram/index.html)) steps through every line of the training loop, with every number worked out.

## What we build

We build a model that gives every word a short list of numbers (an **embedding**). It learns those numbers only from which words appear next to each other. After training, words used in similar places (*cats*, *dogs*, *rabbits*) end up with similar embeddings.

Everything is plain NumPy: a forward pass, the loss, the gradients by hand and the update. No library does the learning for us.

## Intuition

> "You shall know a word by the company it keeps." (J. R. Firth, 1957)

Take the sentence `cats furry pets`. With a window of 1, each word's neighbours are the words right next to it:

| center | context (neighbour) |
|---|---|
| cats | furry |
| furry | cats |
| furry | pets |
| pets | furry |

Skip-gram turns every sentence into these **(center, context)** pairs. It then trains a tiny network to answer one question: *given the center word, which word is its neighbour?*

The network can't answer perfectly, because *furry* sits next to both *cats* and *pets*. Trying anyway forces it to give similar numbers to words that share neighbours. Those numbers are the embeddings.

## The pieces

| Name | Shape | Meaning |
|---|---|---|
| `V` | number | how many different words (vocabulary size); 21 for the toy sentences |
| `D` | number | how many numbers describe each word; 3 here |
| `E` | V × D | the **embeddings**, one row per word. This is what we keep at the end. |
| `W` | D × V | the **output weights**, one column per word, used to score candidate neighbours |
| `h` | D | the center word's row of `E` |
| `scores` | V | one score per word: how well it fits as a neighbour |
| `p` | V | the scores turned into chances that add up to 1 |

## Math

For one pair (center word $c$, context word $x$):

**1. Look up the center word.**

$$h = E_c \qquad (h_i = E_{c,i})$$

**2. Score every word.** Each score is a dot product of $h$ with that word's column of $W$:

$$s_j = \sum_{i=0}^{D-1} h_i \, W_{ij} \qquad\text{for } j = 0,\dots,V-1$$

**3. Turn the scores into chances (softmax).**

$$p_j = \frac{e^{s_j}}{\sum_{k} e^{s_k}}$$

Every $p_j$ is between 0 and 1, and they add up to 1. The code subtracts $\max(s)$ before calling `exp`. This doesn't change the result, but it stops `exp` from overflowing.

**4. Measure how wrong we were (cross-entropy loss).** Only the chance of the real neighbour $x$ counts:

$$L = -\ln p_x$$

If $p_x = 1$ the loss is 0. If $p_x$ is small the loss is large. At the start every $p_j \approx 1/V$, so $L \approx \ln V = \ln 21 \approx 3.04$.

**5. Derive the error at the scores.** We need $\partial L / \partial s_j$. First, the derivative of $-\ln u$:

$$\frac{d}{du}\left(-\ln u\right) = -\frac{1}{u}
\quad\Rightarrow\quad
\frac{\partial L}{\partial p_x} = -\frac{1}{p_x}$$

Next, the derivative of the softmax. Write $p_x = e^{s_x}/Z$ with $Z = \sum_k e^{s_k}$, so $\partial Z/\partial s_j = e^{s_j}$.

- When $j = x$, the quotient rule gives
  $$\frac{\partial p_x}{\partial s_x} = \frac{e^{s_x} Z - e^{s_x} e^{s_x}}{Z^2} = p_x - p_x^2 = p_x(1 - p_x)$$
- When $j \ne x$, only $Z$ depends on $s_j$:
  $$\frac{\partial p_x}{\partial s_j} = -\frac{e^{s_x} e^{s_j}}{Z^2} = -p_x\,p_j$$

Let $y_j = 1$ if $j = x$ and $0$ otherwise. Both cases then fit one formula:

$$\frac{\partial p_x}{\partial s_j} = p_x\,(y_j - p_j)$$

The chain rule puts the two together:

$$\delta_j = \frac{\partial L}{\partial s_j} = -\frac{1}{p_x}\cdot p_x\,(y_j - p_j) = p_j - y_j$$

In words, the error is the chance each word got minus the chance it should have got. In the code this is `error = probabilities.copy(); error[context] -= 1`.

**6. Gradients of the two weight tables.** Since $s_j = \sum_i h_i W_{ij}$, we get $\partial s_j / \partial W_{ij} = h_i$. That gives

$$\frac{\partial L}{\partial W_{ij}} = h_i\,\delta_j
\qquad\Longrightarrow\qquad
\nabla_W L = h \otimes \delta \quad(\text{outer product},\ D\times V)$$

Every score depends on $h_i$ through $W_{ij}$, so we add up over all $j$:

$$\frac{\partial L}{\partial h_i} = \sum_{j} W_{ij}\,\delta_j
\qquad\Longrightarrow\qquad
\nabla_h L = W\,\delta \quad(D)$$

**7. Update with gradient descent.** Take a small step against the gradient. The step size is $\eta$, the learning rate.

$$W \leftarrow W - \eta\,\nabla_W L \qquad E_c \leftarrow E_c - \eta\,\nabla_h L$$

Only the center word's row of $E$ changes. Every column of $W$ changes: the context word's column moves toward $h$, and the other columns move slightly away from it.

## Code walkthrough

The code is in [`skipgram-word2vec.py`](skipgram-word2vec.py). It goes in this order:

1. **Corpus and vocabulary.** It lowercases the sentences, collects the sorted unique words into `vocab`, and builds `word_to_id`.
2. **Hyperparameters.** `WINDOW = 1`, `D = 3`, `LEARNING_RATE = 0.05`, `EPOCHS = 5`.
3. **Pairs.** For every word, it adds every neighbour within `WINDOW` as a (center, context) pair. That makes 108 pairs.
4. **Initialization.** `np.random.seed(42)`, then `E` (V × D) and `W` (D × V) start as small random numbers (× 0.1).
5. **Training loop.** For each epoch it shuffles the pairs, then for each pair runs steps 1–7 above: `h`, `scores`, `softmax`, `loss`, `error`, `gradient_W`, `gradient_h`, and the two updates.
6. **Results.** It prints the learned embeddings, plots the loss, and plots the embeddings in 2-D with PCA (`plot_embeddings`). It then lists the most similar words using `cosine_similarity`.

## Run it

From the repository root:

```bash
source .venv/bin/activate
cd chapters/01-word2vec-skipgram
python skipgram-word2vec.py
```

Each run writes a log to a new `skipgram-word2vec.output_N.txt` next to the script. These logs are gitignored. Older logs are kept in `outputs/`.

## What to notice

- With the default settings, the loss only falls from 3.04 to about 3.00 in 5 epochs: the learning rate is small. In the walkthrough page, the preset with `LEARNING_RATE = 0.3` and 10 epochs gets it down to about 1.45.
- The loss can never reach 0. *furry* really does sit next to *cats*, *dogs*, *rabbits* and *pets*, so no model can be sure which one it will see. The best possible average here is about 1.17.

## Exercises

1. Set `D = 2` and compare the embedding plot with the PCA plot for `D = 3`.
2. Change `WINDOW` to 2. How many pairs are there now, and why?
3. Try `LEARNING_RATE = 1.0`. What happens to the loss?
4. Time one epoch with V = 21. Then imagine V = 50,000: which line of the loop would dominate? (Chapter 02 answers this.)

## What's next

**02 · Negative sampling.** Every update above touches all V columns of `W` and computes a softmax over the whole vocabulary. That is fine for 21 words, but far too slow for 50,000. Negative sampling replaces the softmax with a handful of yes/no questions: "is this word a real neighbour?" for the real one, and for a few random ones.
