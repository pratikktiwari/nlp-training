# nlp-training

A step-by-step path from word embeddings to GPT-2, built from scratch. Each chapter adds one idea on top of the previous one and comes in three parts:

- **Concepts**: the chapter's `README.md`. It covers the intuition first, then the math with every step derived, then a walk through the code.
- **Code**: small, runnable scripts. NumPy first so every gradient is written by hand; PyTorch once the math is clear.
- **Visualization**: an interactive page that replays a real training run line by line. All pages are published from [`docs/`](docs/) with GitHub Pages.

## Roadmap

| # | Chapter | What it adds | Status |
|---|---|---|---|
| 01 | [Word2Vec: skip-gram](chapters/01-word2vec-skipgram/) | embeddings, softmax, cross-entropy, backprop by hand | ✅ code · ✅ viz · ✅ notes |
| 02 | [Negative sampling](chapters/02-negative-sampling/) | why full softmax is slow; sigmoid plus k "wrong" words | ✅ code · ✅ viz · notes next |
<!-- | 03 | Bigram language model | the first model that predicts the next token | planned |
| 04 | MLP language model | a context window, batches, train/validation split | planned |
| 05 | RNN language model | a hidden state carried through time; backprop through time | planned |
| 06 | LSTM and GRU | gates that keep gradients from vanishing (switch to PyTorch) | planned |
| 07 | Autograd | build a tiny autograd engine, then use PyTorch's | planned |
| 08 | Attention | seq2seq plus attention; query, key and value | planned |
| 09 | Transformer block | self-attention, multi-head, LayerNorm, residuals, positions | planned |
| 10 | BPE tokenizer | byte-pair encoding, the tokenizer GPT-2 uses | planned |
| 11 | Mini GPT | a small character-level GPT | planned |
| 12 | GPT-2 (124M) | reproduce GPT-2 small and compare with OpenAI's weights | planned |
| 13+ | Beyond | fine-tuning, LoRA, KV cache, sampling, RLHF/DPO | planned | -->

## Layout

```
chapters/NN-name/     README.md (concepts), code, raw/ (pure-Python versions), outputs/ (gitignored)
docs/NN-name/         index.html, the interactive visualization for that chapter
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

In the repository's **Settings → Pages**, set the source to the main branch and the `/docs` folder. The landing page is `docs/index.html`, and each chapter's visualization is served at `/<chapter-folder>/`.
