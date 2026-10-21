# 03 · GloVe

**The idea:** collect how often words occur near one another, then learn vectors and biases that reconstruct the logarithms of those counts.

Chapters [01](../01-word2vec-skipgram/README.md) and [02](../02-negative-sampling/README.md) introduced embeddings, dot products, losses, and gradients. This chapter changes the task from predicting context identities or distinguishing data from noise to **weighted regression on corpus statistics**.

**Use this chapter in order:** follow one count into its prediction and loss, understand how shared parameters receive feedback, then inspect repeated full-batch updates.

- [Code](glove.py)
- [Pure-Python code (no NumPy or PyTorch)](raw/glove.py)
- [GloVe intuition: from context counts to learned vectors](https://pratikktiwari.github.io/text-nn-from-scratch/03-glove/intuition.html) ([page source](../../docs/03-glove/intuition.html))
- [Interactive math and PyTorch walkthrough](https://pratikktiwari.github.io/text-nn-from-scratch/03-glove/) ([page source](../../docs/03-glove/index.html))
- [Previous chapter: negative sampling](../02-negative-sampling/README.md)
- [Next chapter: n-gram language model](../04-ngram-language-model/README.md)

## What we build

A small PyTorch GloVe model using the supplied reference's corpus, initialization, weighted log-count objective, full-batch autograd, and Adagrad optimizer. The gradients are derived below but calculated by PyTorch in the script.

| Setting | Default |
|---|---|
| Corpus | 10 sentences about ice, water, steam, cats, and dogs |
| Vocabulary `V` | 21 words, sorted alphabetically |
| Context window | 2 positions on either side, within a sentence |
| Contribution per context occurrence | `1 / token_distance` |
| Training batch `M` | 80 unique nonzero directed cells |
| Embedding dimension `D` | 20 per word per role |
| Parameters | Two `21 × 20` vector tables and two `21 × 1` bias tables: 882 scalars |
| Training | 300 full-batch updates, seed 42, CPU float32 |
| Optimizer | Adagrad, learning rate 0.03 |
| Final representation | Target vector + context vector |

This is a **different corpus** from the pets/fruit/vehicles sentences in chapters 01 and 02. It does not apply sigmoid, softmax, negative sampling, or next-word generation.

## Intuition

### 1. Build the evidence before updating any vectors

Word IDs identify words but supply no useful geometry. We do not know a correct vector for `ice`; we do know its observed contexts. A reconstruction task gives adjustable vectors feedback from that evidence.

Count a directed pair whenever its two positions fall inside the window. Close occurrences contribute more: distance 1 adds 1; distance 2 adds 0.5. Repeated occurrences add to the same table cell.

For `ice / water`:

- `"ice melts in water"` contributes nothing: the distance is 3.
- `"ice and water are related"` contributes `1/2`.
- Therefore `X[ice, water] = 0.5`. The reverse direction is counted separately.

These are weighted counts, not probabilities. “Global” means aggregated statistics across this corpus, not a context window spanning every sentence. A zero is unobserved under this window, **not a negative example**. We fit only nonzero cells.

### 2. Predict a log count rather than a raw count

Raw counts can span a large range. Log compresses that range and makes equal ratios produce equal differences: `log(2X) - log(X) = log(2)`.

For the actual observed counts:

| `X` | Target `log(X)` |
|---|---|
| 0.5 | about -0.693147 |
| 1 | 0 |
| 2 | about +0.693147 |

A negative target is valid. Since `exp(-0.693147) ≈ 0.5`, this target represents a fractional weighted count, not a negative probability. Missing cells have no target; we do not calculate `log(0)` or replace them with zero targets.

### 3. Let shared vectors and biases reconstruct those targets

Both word IDs are inputs. The unknown is their log count, not the context ID. Select the target vector, select the context vector, multiply matching coordinates, and sum. Add a target bias and a context bias.

The dot product learns interactions between roles. Biases supply broad row/column offsets, so vectors need not carry every baseline effect. Without biases, the model is still a regression model, but has fewer ways to separate offsets from interactions. Biases are learned scalars, not assigned frequency logarithms.

A word uses the same target row across several contexts. Adjusting it can improve one cell while worsening another. This sharing is why we need the full objective, rather than considering one pair in isolation.

### 4. Measure mistakes, then assign their influence

The residual is prediction minus target. Squaring penalizes mistakes on either side and supplies a smooth slope. Predicting twice an observed count gives log residual `log(2)`; predicting half gives `-log(2)`. Their squared costs are equal.

A separate function `f(X)` assigns less influence to lower-count observations and caps the influence of frequent ones. This is a heuristic confidence weighting, **not** the inverse-distance rule that built the table.

For `ice / water`, `f(0.5) ≈ 0.105737`. If the prediction were exactly 0, its residual would be `+0.693147`, giving weighted cost about `0.0508017`, or `0.000635022` after dividing by 80. The initialized model is near, but not exactly at, zero; the HTML shows its actual values.

The loss chooses what “better” means. Gradients then measure how parameters affect that loss; Adagrad uses these gradients and its history to choose actual changes. Neither a forward pass nor `backward()` alone updates a weight.

## Math

**Notation:** `log` means base e, $\log_e$, matching `torch.log`. Here $w_i,u_j$ are separate target/context vectors, $b_i,c_j$ are their scalar biases, and $\Omega$ contains the nonzero directed cells. $M=|\Omega|=80$.

### 1. Forward pass

First combine the two roles into an unrestricted prediction, then compare it with the fixed log count:

$$y_{ij}=w_i^\top u_j+b_i+c_j,\qquad
t_{ij}=\log X_{ij},\qquad r_{ij}=y_{ij}-t_{ij}.$$

There is no probability transformation. A prediction near 0 reconstructs a count near $\exp(0)=1$, not “no occurrence.”

Weight and average the squared residuals:

$$f(x)=
\begin{cases}
(x/10)^{0.75}, & x<10\\
1, & x\geq10
\end{cases}
\qquad
L=\frac{1}{M}\sum_{(i,j)\in\Omega} f(X_{ij})r_{ij}^{2}.$$

All observed counts here are below 10, so the cap is not reached. Counts, targets, and weights stay fixed during training. The denominator is the number of observed cells, not the sum of their weights or all 441 matrix cells.

The commonly written GloVe objective uses a sum; this reference uses `.mean()`. Keep the resulting `1/M` in every derivative.

### 2. Backward pass

To decide how changing a prediction affects the mean loss, differentiate its squared residual:

$$a_{ij}=\frac{\partial L}{\partial y_{ij}}
=\frac{2f(X_{ij})r_{ij}}{M}.$$

Overprediction gives a positive derivative; lowering the prediction is locally helpful. Underprediction gives a negative derivative; raising it is locally helpful. The interactive curve shows this sign as the tangent's slope.

The chain rule connects that signal to parameters. A bias changes its prediction one-for-one. A target coordinate changes it by the matching context coordinate. Add contributions from every cell that reuses the parameter:

$$\frac{\partial L}{\partial w_i}=\sum_{j:(i,j)\in\Omega}a_{ij}u_j,
\qquad
\frac{\partial L}{\partial u_j}=\sum_{i:(i,j)\in\Omega}a_{ij}w_i,$$

$$\frac{\partial L}{\partial b_i}=\sum_{j:(i,j)\in\Omega}a_{ij},
\qquad
\frac{\partial L}{\partial c_j}=\sum_{i:(i,j)\in\Omega}a_{ij}.$$

Thus a selected pair's contribution is not generally the full gradient of its vector or bias. The walkthrough displays both that contribution and the actual full-batch autograd gradient. Small one-parameter probes introduce this sensitivity before the formal derivative.

### 3. Adagrad update

For **each scalar** parameter $\theta$, use its full-batch gradient $g$ and its own accumulated squared-gradient history $A$:

$$A_{\mathrm{new}}=A_{\mathrm{old}}+g^2,\qquad
\theta_{\mathrm{new}}=\theta_{\mathrm{old}}-
\frac{0.03g}{\sqrt{A_{\mathrm{new}}}+10^{-10}}.$$

Subtracting the gradient moves in a locally decreasing direction; the denominator adapts the scale using that parameter's history. The accumulator starts at 0; epsilon is added **after** the square root. With a first nonzero gradient, the step can be almost 0.03 in magnitude because $\sqrt{g^2}=|g|$.

This run uses no learning-rate decay or weight decay. `zero_grad()` clears `.grad`, not Adagrad's accumulator. One epoch calculates all 80 cells before taking **one** optimizer step.

### 4. Extract representations for comparison

The query helpers use:

$$e_i=w_i+u_i,\qquad
\operatorname{cosine}(e_i,e_j)=
\frac{e_i^\top e_j}{\|e_i\|\|e_j\|+10^{-10}}.$$

This combines the two roles for a chosen downstream comparison. Count prediction still uses $w_i^\top u_j+b_i+c_j$, **not** $e_i^\top e_j$. Low regression loss therefore does not directly imply desirable cosine rankings.

## Code walkthrough

1. **`tokenize` / `prepare_corpus`:** lowercase regex tokenization, sorted vocabulary, sentence-local windows, and accumulated inverse-distance counts. The dense matrix is for inspection; aligned `target_ids`, `context_ids`, and `counts` contain only observed cells. Different positions containing the same word can form a diagonal cell.
2. **`GloVe`:** four `nn.Embedding` tables. Vector coordinates start uniformly in `[-0.025, 0.025]`, and biases start at zero. Constructing tables consumes random numbers before custom initialization; preserving that order matters for reference parity.
3. **`forward`:** look up all rows, compute elementwise products and `.sum(dim=1)`, squeeze scalar bias rows, and add. The result has one prediction per observed cell.
4. **`train_glove`:** prepare fixed `torch.log(counts)` and confidence weights. Each loop computes the mean loss, clears old gradients, calls `loss.backward()`, and then `optimizer.step()`.
5. **Recording:** copies of weights, predictions, cell losses, autograd gradients, and actual Adagrad accumulators preserve the real training path. Recording does not replace the optimizer with a hand-written update.
6. **Query helpers:** add the two vector tables for embeddings and cosine queries. `predict_log_count` instead calls the original two-role model with biases.
7. **`main`:** prints the data, loss checkpoints, embeddings, rankings, and count inspection. Numbered `glove.output_N.txt` logs sit beside the script and are gitignored.

## Run it

From the repository root, with the project's virtual environment active:

```bash
pip install -r requirements.txt
python chapters/03-glove/glove.py
python -m unittest discover -s chapters/03-glove -p test_glove.py -v
```

### Pure-Python version

The standalone [`raw/glove.py`](raw/glove.py) uses only the Python standard
library. No dependency installation is needed:

```bash
python3 chapters/03-glove/raw/glove.py
```

It uses the same corpus, distance-weighted counts, 20-dimensional vectors,
weighted mean squared log-count loss, and 300 full-batch Adagrad updates.
`loss_and_gradients` explicitly accumulates all vector and bias gradients
before updating any parameters; the update explicitly maintains the sum of
squared gradients and adds `1e-10` outside the square root.
It records parameter, gradient, accumulator, prediction, and loss histories,
and provides the same embedding, cosine, and count-prediction queries.
Numbered output logs are saved beside the raw script.

Python's `random.Random(42)` and ordinary Python floats replace PyTorch's
random generator and float32 tensors. The initialization range and seed are
the same, but the exact initial vectors, training trajectory, and cosine
rankings differ. This version does not export or replace the PyTorch browser
recording.

If Node.js is available, validate the browser replay's calculations against every recorded PyTorch update:

```bash
node --test chapters/03-glove/test-replay.cjs
```

This chapter adds PyTorch; chapters 01 and 02 still use NumPy and manual gradients. The training runs on CPU and needs no GPU or downloads of pretrained models.

To regenerate the default browser recording and serve the site:

```bash
python chapters/03-glove/glove.py --export-web
python -m http.server 8765 --directory docs
```

Open http://localhost:8765/03-glove/intuition.html or http://localhost:8765/03-glove/. Use HTTP/localhost or HTTPS, not `file://`: the replay fetches data and verifies SHA-256 hashes.

### What the browser does and does not run

The bundled [`training.json`](../../docs/03-glove/training.json) manifest and [`training.f32`](../../docs/03-glove/training.f32) tensors contain all 301 parameter states and 300 updates. The manifest includes the exact Python source, operation-to-line mappings, run settings, and source/binary hashes. Regenerate both files whenever the source changes.

The browser **replays recorded PyTorch results**; it does not run PyTorch/autograd or silently substitute JavaScript training. Auxiliary decompositions, sensitivity probes, and curves use JavaScript arithmetic on the float32 recording, so very small rounding differences are normal.

Both GloVe pages typeset their equations with MathJax, including the worked
calculations refreshed by the replay controls. The browser needs access to
the MathJax CDN; Python source and API calls remain displayed as code.

Controls select checkpoints, cells, vector coordinates, and parameters; they do not change training hyperparameters. For experiments, edit the Python constants and rerun the script. The published lesson's prose and worked defaults describe the supplied reference settings; publishing a different recording also requires updating its static examples and labels.

### What to notice

Measured default-run checkpoints:

| Completed updates | Full-batch mean loss |
|---|---|
| 0 | 0.0237101894 |
| 1 | 0.0200573392 |
| 10 | 0.0007613324 |
| 20 | 0.00000376976 |
| 50 | about `5.02e-10` |
| 300 | about `1.67e-17` |

The reference logs loss **before** each update. Its “epoch 50” value is checkpoint 49, not 50. This script labels that convention and also prints the final post-update loss.

The tiny numerical floor is not exact zero, and late updates need not improve monotonically. Versions/hardware can cause small differences. Do not compare this objective's raw values with either earlier chapter's classification loss.

**A good count fit is not a semantic guarantee.** In the supplied run, final cosines are approximately `ice / water = -0.7507`, `ice / steam = 0.3048`, and `cats / dogs = 0.7455`. There are 882 parameters but only 80 targets, with missing cells unconstrained. Many settings can fit this toy data; a negative cosine is not a statement that ice and water are unrelated.

## Exercises

Change one thing at a time in Python, and compare the appropriate objective and queries:

1. Find all distance contributions for `ice / water` with window 2. Predict which extra occurrence window 3 includes before rebuilding the table.
2. Inspect a cell with count 1. Its log target is zero. Explain why this differs from an unobserved cell with count zero.
3. At checkpoint 0, choose the target bias for `ice`. Sum every affected cell's gradient contribution. Why is the selected pair alone insufficient?
4. Compare the first and a later Adagrad update for one coordinate. Use its gradient and accumulator to reconstruct the actual step, rather than `-lr * gradient`.
5. Set the learning rate to zero and rerun Python. Check that loss and parameters stay unchanged. Restore 0.03 afterwards.
6. Compare `D = 2` and `D = 20`. Look at fitted counts and cosine rankings separately; a smaller training loss alone does not decide which representation is useful.

## What's next

Compare three distinct ways to obtain learning feedback: full-softmax context classification, data-versus-noise decisions, and weighted log-count reconstruction. Identify what each predicts and penalizes before comparing its embeddings.

For this chapter, the important boundary is between **fitting observed corpus statistics** and **evaluating representations**. More text and held-out evaluation are needed before making claims about useful semantics.
