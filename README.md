# text-nn-from-scratch

**Live site:** https://pratikktiwari.github.io/text-nn-from-scratch/

Small implementations, explanations, and interactive experiments for learning neural networks for text.

- **Intuition**: a separate page for each chapter explains the training task, predictions, choice of loss, and effect of gradient updates. Play a deterministic training run to see predictions and measured loss change together.
- **Concepts**: the chapter's `README.md`. It covers the intuition first, then the math with every step derived, then a walk through the code.
- **Code**: small, runnable NumPy scripts with gradients written by hand.
- **Visualization**: an interactive page that replays a real training run line by line. All pages are published from [`docs/`](docs/) with GitHub Pages at [https://pratikktiwari.github.io/text-nn-from-scratch/](https://pratikktiwari.github.io/text-nn-from-scratch/).

## How to learn with a chapter

Start with [Word2Vec intuition](docs/01-word2vec-skipgram/intuition.html) or [negative sampling intuition](docs/02-negative-sampling/intuition.html). Each follows one example from embedding lookup through dot products, output transformations, likelihood-based loss, and gradients. Small weight probes show what sensitivity means before introducing an update. These are separate pages, not modes in one sandbox.

The final experiment replays 200 epochs using the chapter corpus, seed, dimensions, window, and sampling rules. The longer run makes loss trends visible; the scripts default to 5 and 50 epochs respectively. Set the learning rate to zero to test what happens without updates. The negative-sampling plot also includes a constant-output baseline that fits only the 1:3 label ratio, to distinguish that shortcut from learning word-dependent behaviour.

Then open the chapter's math-and-code walkthrough, choose an experiment, and follow the highlighted code. Each training step links to a synchronized "Why this step?" explanation before its existing equations, derivation, and numerical tables. The explanations distinguish predictions, loss, gradients, and parameter updates, and use the selected run's values.

Change one setting at a time and predict what it should affect before rerunning. Finish with the chapter's takeaways: lower loss means fitting that chapter's objective better, not necessarily better word similarity. In particular, full-softmax and negative-sampling losses are not directly comparable.

The site defaults to light mode. The theme switch on the homepage and each learning page remembers an explicit choice across navigation and reloads. Mathematical `log` notation means base e, matching NumPy. In the derivations, synchronized curves show the current value and a dashed tangent: its slope explains the sign and size of the error or gradient without changing the training run.

## Roadmap

| # | Chapter | What it adds | Status |
|---|---|---|---|
| 01 | [Word2Vec: skip-gram](chapters/01-word2vec-skipgram/) | embeddings, softmax, cross-entropy, backprop by hand |
| 02 | [Negative sampling](chapters/02-negative-sampling/) | why full softmax is slow; sigmoid plus k "wrong" words |
<!-- | 03 | Bigram language model | the first model that predicts the next token |
| 04 | MLP language model | a context window, batches, train/validation split |
| 05 | RNN language model | a hidden state carried through time; backprop through time |
| 06 | LSTM and GRU | gates that keep gradients from vanishing (switch to PyTorch) |
| 07 | Autograd | build a tiny autograd engine, then use PyTorch's |
| 08 | Attention | seq2seq plus attention; query, key and value |
| 09 | Transformer block | self-attention, multi-head, LayerNorm, residuals, positions |
| 10 | BPE tokenizer | byte-pair encoding, the tokenizer GPT-2 uses |
| 11 | Mini GPT | a small character-level GPT |
| 12 | GPT-2 (124M) | reproduce GPT-2 small and compare with OpenAI's weights |
| 13+ | Beyond | fine-tuning, LoRA, KV cache, sampling, RLHF/DPO | -->

## Layout

```
chapters/NN-name/     README.md (concepts), code, raw/ (pure-Python versions), outputs/ (gitignored)
docs/NN-name/         intuition.html (why it learns), index.html (math and code)
docs/index.html       landing page for GitHub Pages
common/               helpers shared by 3+ chapters (empty until needed)
archive/              older notebooks, not part of the path
```

## Conventions

1. **Every chapter runs on its own.** Copy a small helper rather than importing it from another chapter. It moves to `common/` once three chapters repeat it.
2. **Every chapter README has the same outline:** What we build · Intuition · Math · Code walkthrough · Run it · Exercises · What's next.
3. **Seed 42 everywhere**, so the visualizations can replay the exact run.
4. **The same toy sentences** (pets, fruit, vehicles) carry through the early chapters, so results can be compared, before we move to real text.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cd chapters/01-word2vec-skipgram
python skipgram-word2vec.py
```

## GitHub Pages

In the repository's **Settings → Pages**, set the source to the main branch and the `/docs` folder. The landing page is `docs/index.html`, and each chapter's visualization is served at `/<chapter-folder>/`, for example [https://pratikktiwari.github.io/text-nn-from-scratch/01-word2vec-skipgram/](https://pratikktiwari.github.io/text-nn-from-scratch/01-word2vec-skipgram/) and [https://pratikktiwari.github.io/text-nn-from-scratch/02-negative-sampling/](https://pratikktiwari.github.io/text-nn-from-scratch/02-negative-sampling/).
