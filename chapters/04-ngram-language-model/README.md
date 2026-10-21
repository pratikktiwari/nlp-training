# 04 · N-gram language model

**The idea:** read the last n − 1 words and guess the next one. Do it once by counting, and once with a small neural network, both written from scratch in NumPy.

Chapters [01](../01-word2vec-skipgram/README.md) to [03](../03-glove/README.md) learned word vectors. This chapter uses words to **predict the next word**, which is what a language model does. Then it uses those predictions to write text.

**Use this chapter in order:** see how a window of text becomes a training example, follow the count model, then the neural model, then read the generated text.

- [Code](ngram-language-model.py)
- [Pure-Python code (no NumPy or third-party dependencies)](raw/ngram-language-model.py)
- [N-gram intuition: predicting the next word](https://pratikktiwari.github.io/text-nn-from-scratch/04-ngram-language-model/intuition.html) ([page source](../../docs/04-ngram-language-model/intuition.html))
- [Interactive math and code walkthrough](https://pratikktiwari.github.io/text-nn-from-scratch/04-ngram-language-model/) ([page source](../../docs/04-ngram-language-model/index.html))
- [Generated text from the last run](generated-text.md)
- [Previous chapter: GloVe](../03-glove/README.md)

## What we build

Two next-word models, trained and tested on the same text:

| | Count model | Neural model |
|---|---|---|
| What it stores | For every context, how often each word came next | An embedding table `E`, an output table `W` and a starting score `b` per word |
| How it guesses | The word that came next most often | Softmax over a score for every word |
| How it learns | Counting, once | Gradient descent with hand-written gradients |
| Context it never saw | Drops the oldest word and looks again ("backs off") | Still gives scores, because similar words have similar embeddings |

| Setting | Default |
|---|---|
| Text | WikiText-2: Wikipedia articles, about 2.0 million training words |
| Vocabulary `V` | 10,000: the 9,999 most common training words plus `<unk>` for every other word |
| `n` | 2, 3 and 4 (read 1, 2 or 3 previous words) |
| Embedding size `D` | 64 |
| Training | 1 epoch, batches of 512 windows, learning rate 4.0, seed 42 |
| Generation | 60 words after each of 3 prompts, greedy and sampled |

Everything is NumPy, like chapters 01 and 02. PyTorch comes later, once we have built the pieces by hand.

## Intuition

### 1. Slide a window over the text

Take `the cat sat on the mat`. With n = 3, each window is 3 words in a row. The first 2 are the **context** (what the model reads), the last one is the **target** (what it must guess):

| Context | Target |
|---|---|
| the cat | sat |
| cat sat | on |
| sat on | the |
| on the | mat |

Each window is one training example, and we get one per word of text. `<eos>` ("end of sentence") is added after every paragraph and treated as a normal word, so the model also learns where text stops.

### 2. Count model: remember what came next

For the context `on the`, find every place in the training text where `on the` appears and count the word after it. Divide each count by the total to get a probability:

$$P(\text{next} = w \mid \text{context}) = \frac{\text{count}(\text{context},\ w)}{\text{count}(\text{context})}$$

The guess is the word with the biggest count. This needs no training loop at all.

**When the context never appeared**, both counts are 0 and the fraction is 0 / 0. The script then **backs off**: it drops the oldest context word and looks up the shorter context. With no context left, it uses plain word counts, which always exist.

### 3. Bigger n: better guesses, but rarer contexts

After `the`, almost anything can come next. After `the first season of the`, very few words fit. So more context should help.

But each extra word makes the context rarer. At n = 4, most 3-word contexts in new text never appeared in the training text, so the model backs off and the extra context is wasted. Meanwhile, accuracy on the **training** text shoots up, because the model is memorising whole phrases. That gap is the main lesson of the n table below.

### 4. Neural model: similar words share what they learned

The count model treats `cat` and `dog` as unrelated keys. Seeing `the cat sat` teaches it nothing about `the dog sat`.

The neural model looks up an embedding for each context word, joins them into one list, and scores every vocabulary word with one linear layer, like chapter 01. Since `cat` and `dog` appear in similar places, their embeddings become similar, so they produce similar scores. Every context gets scores, even one never seen whole.

### 5. Generate text by feeding guesses back in

Start with a prompt, predict the next word, append it, and repeat with the last n − 1 words.

- **Greedy** takes the most likely word every time. Because the model only sees the last n − 1 words, the same context always gives the same word. Once a context repeats, the text after it repeats too, forever: `the first time, and the first time, and the first time, ...`
- **Sampled** draws the next word at random by its probability. That breaks the loops, but the text wanders, because nothing reminds the model what it said more than n − 1 words ago.

That short memory is the classic limit of n-gram models. Later chapters (RNN, LSTM, attention) remove it.

## Math

**Notation:** `ln` is the natural logarithm, matching `np.log`. $V$ words, embedding size $D$, context length $C = n - 1$. $E$ is $V \times D$, $W$ is $CD \times V$, $b$ has $V$ numbers. Context word IDs $w_1, \dots, w_C$, target ID $t$.

### Forward pass

Join the context embeddings end to end:

$$x = [\,E_{w_1},\ E_{w_2},\ \dots,\ E_{w_C}\,] \qquad (CD \text{ numbers})$$

Score every word, turn scores into probabilities, and measure the loss:

$$s_v = \sum_{i=1}^{CD} x_i W_{i,v} + b_v \qquad p_v = \frac{e^{s_v}}{\sum_u e^{s_u}} \qquad L = -\ln p_t$$

Why join and not add? Adding `E[cat] + E[sat]` would give the same `x` for `cat sat` and `sat cat`. Joining keeps each word in its own slot, so word order matters.

Why `b`? It gives each word a starting score that does not depend on the context. Common words such as `the` and `,` can be likely after almost anything; `b` lets the model learn that once, instead of through every context.

### Backward pass

Softmax with cross-entropy gives the same score gradient as chapter 01, "probability minus 1 for the target":

$$g_v = \frac{\partial L}{\partial s_v} = p_v - [v = t]$$

Each score is a sum of products $x_i W_{i,v}$, plus $b_v$. The chain rule gives:

$$\frac{\partial L}{\partial W_{i,v}} = x_i\, g_v \qquad \frac{\partial L}{\partial b_v} = g_v \qquad \frac{\partial L}{\partial x_i} = \sum_v W_{i,v}\, g_v$$

`x` was made by copying rows of `E`, so its gradient is cut back into `C` pieces of `D` numbers. Piece `c` is the gradient for row $E_{w_c}$. If one word appears twice in a context, both pieces are added to its row.

### Update and batches

$$W \leftarrow W - \eta\,\frac{\partial L}{\partial W} \qquad b \leftarrow b - \eta\,\frac{\partial L}{\partial b} \qquad E_{w_c} \leftarrow E_{w_c} - \eta\,\Big(\frac{\partial L}{\partial x}\Big)_{\text{piece } c}$$

The script trains on a batch of `B = 512` windows at once. The loss is the batch **mean**, so every $g_v$ is divided by `B`, and the gradients of `W` and `b` are summed over the batch. That is why the learning rate (4.0) is much bigger than chapter 01's 0.05, which updated after every single pair.

### Accuracy

$$\text{accuracy} = \frac{\text{number of windows where the top-scoring word is the real next word}}{\text{number of windows}}$$

`<unk>` counts like any other word here, for both models. About 11% of test words are `<unk>`.

## Code walkthrough

Read [`ngram-language-model.py`](ngram-language-model.py) in these blocks:

| Block | What to look for |
|---|---|
| 1. Load the text | `download_wikitext` saves the 3 files to `data/` once. `read_tokens` lowercases, skips headings, adds `<eos>`. `build_vocab` keeps the 9,999 most common words. `make_windows` slides the window with NumPy. |
| 2. Count model | `train_count_model` counts `(context, next word)` with `Counter` for every context length, keeps the top word. `count_predict` backs off. `count_probabilities` searches the training windows for generation. |
| 3. Neural model | `init_neural_model` (E, W, b). `forward` looks up and joins. `train_step` has the forward pass, the hand-written gradients and the update. `np.add.at` handles a word that appears twice in a batch. |
| 4. Generate text | `generate` repeats predict, append, slide. Greedy uses `argmax`; sampling uses `rng.choice` with the probabilities. `<unk>` is never written. |
| 5. Run everything | For each n: build the count model, train the neural model, time both with `time.perf_counter`, test, and generate. |

## Run it

From the repository root, with the virtual environment active:

```bash
python chapters/04-ngram-language-model/ngram-language-model.py
python -m unittest discover -s chapters/04-ngram-language-model -p test_ngram.py -v
```

The first run downloads WikiText-2 (about 13 MB) from the PyTorch examples repository into `chapters/04-ngram-language-model/data/`, which is gitignored. Training the neural model takes a few minutes for each n on a laptop CPU; the count model takes seconds.

The script prints accuracy and timings, and writes:

- `ngram-language-model.output_N.txt`, the full log (gitignored, never overwritten).
- [`generated-text.md`](generated-text.md), all 36 generated paragraphs (3 values of n × 2 models × greedy/sampled × 3 prompts). This file is meant to be committed.

To refresh the numbers on the web pages, run with `--export-web`. It writes `docs/04-ngram-language-model/results.json`.

The tests check the window shapes, the back-off, that the hand-written gradients match finite differences, and that generation never writes `<unk>`.

### Pure-Python version

[`raw/ngram-language-model.py`](raw/ngram-language-model.py) implements both
models with Python lists, counters, `math`, and `random`. No packages need
installing:

```bash
python3 chapters/04-ngram-language-model/raw/ngram-language-model.py
```

The default is a practical demo using the browser lesson's ten training
sentences and separate held-out validation/test sentences. It runs n = 2, 3,
and 4 with 8-dimensional embeddings, 100 epochs, batches of 8, learning rate
0.5, and seed 42. It reports accuracy, full-context coverage, loss, timings,
and all 36 greedy/sampled continuations. These demo metrics are not the
WikiText-2 results in the table below.

The count model stores next-word counts for each context length and backs
off by dropping the oldest word. The neural model concatenates embeddings,
computes stable softmax and cross-entropy, and accumulates manual batch
gradients before changing any weights. Repeated context word IDs add their
embedding gradients into the same row. Generation excludes `<unk>`.

For a small WikiText-2 run:

```bash
python3 chapters/04-ngram-language-model/raw/ngram-language-model.py \
  --wikitext --max-tokens 1000 --vocab-size 100 --dimensions 8 \
  --batch-size 8 --learning-rate 0.5
```

`--wikitext` alone selects the original script's settings: vocabulary cap
10,000, 64 dimensions, 1 epoch, batch size 512, and learning rate 4.0.
**Full WikiText-2 neural training is extremely slow with Python loops.**
Use the limits above for exploration, or the NumPy implementation for the
full dataset. The smaller-batch example also lowers the learning rate to
avoid unstable updates. Downloads share the chapter's existing `data/` cache.
Additional options include `--epochs`, `--learning-rate`, and
`--generate-words`; see `--help`.

The raw script saves results and every generated paragraph to its own
numbered `raw/ngram-language-model.output_N.txt` log (gitignored). It does
not overwrite `generated-text.md` or export the website's results.
Python's random generator and ordinary floats differ from NumPy's generator
and float32 arrays, so exact weights, metrics, and samples will differ.

## What to notice

Measured on the default run (your timings will differ):

| n | Count: full context seen in training | Count: accuracy on training text | Count: accuracy on test text | Neural: accuracy on test text | Neural: training time |
|---|---|---|---|---|---|
| 2 | 100% | 19.7% | 19.1% | 17.8% | about 4 min |
| 3 | 86.5% | 45.3% | 20.5% | 18.8% | about 4.5 min |
| 4 | 51.4% | 76.3% | 19.6% | 19.4% | about 5 min |

Training accuracy uses the first 200,000 training windows. Times are from a 4-core cloud machine; building each count model took 4 to 12 seconds.

- **The count model memorises.** From n = 2 to n = 4 its training accuracy goes from 20% to 76%, but its test accuracy stays near 20% and is best at n = 3. At n = 4, only half of the test contexts were ever seen in training.
- **The neural model improves with n, but does not catch up after 1 epoch.** It gains from more context (17.8% → 19.4%) because it never has to back off, but one pass of plain gradient descent is not much training. It is also far slower than counting: about 4,000 batches of matrix products against one counting pass.
- **Greedy text loops.** Look for the repeated phrases in `generated-text.md`. Sampled text does not loop, but it does not stay on one topic.
- **Bigger n makes greedy text look more fluent** for the count model, because it copies longer phrases from Wikipedia. Check whether those phrases appear word for word in `data/wiki.train.txt`.

## Exercises

Change one thing at a time:

1. Set `EPOCHS = 3`. Does the neural model now beat the count model? How does the training time change?
2. Add `n = 5` to `N_VALUES`. Predict the count model's training and test accuracy before running.
3. In `generate`, divide the scores by a "temperature" before softmax: try 0.7 and 1.5. Which one reads better, and which one loops?
4. Set `VOCAB_SIZE = 5_000`. Accuracy goes up. Why is that partly misleading? (Hint: how often is `<unk>` the answer?)
5. Replace the count model's back-off with add-one smoothing: $(\text{count}+1)/(\text{count}(\text{context})+V)$. What happens to the guess for an unseen context?

## What's next

Both models forget everything older than n − 1 words. Next we keep the neural model but give it a hidden layer (an MLP language model), and then a memory that runs along the whole text (an RNN).
