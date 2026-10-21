// The chapter 04 models, small enough to run in the browser.
// Same steps as ngram-language-model.py, on a 10-sentence corpus instead of WikiText-2.
const NgramEngine = (() => {
  const CORPUS = [
    'the cat sat on the mat',
    'the dog sat on the rug',
    'the cat ate the fish',
    'the dog ate the bone',
    'a cat sat on a chair',
    'a dog ran in the park',
    'the bird sat on the tree',
    'the bird ate the seed',
    'a cat ran to the door',
    'the dog ran to the door',
  ];
  const D = 2;
  const LEARNING_RATE = 0.5;
  const BATCH_SIZE = 8;

  // one long list of words; every sentence ends with <eos>, like the Python script's paragraphs
  const tokens = CORPUS.flatMap(sentence => [...sentence.split(' '), '<eos>']);

  // most common word first; ties keep the order of first appearance (Python's Counter.most_common does the same)
  const counts = new Map();
  for (const word of tokens) counts.set(word, (counts.get(word) || 0) + 1);
  const vocab = [...counts.keys()].sort((a, b) => counts.get(b) - counts.get(a));
  const wordToId = new Map(vocab.map((word, i) => [word, i]));
  const ids = tokens.map(word => wordToId.get(word));
  const V = vocab.length;

  function windows(n) {
    const result = [];
    for (let i = 0; i + n <= ids.length; i++) {
      result.push({ position: i, context: ids.slice(i, i + n - 1), target: ids[i + n - 1] });
    }
    return result;
  }

  // ---------- count model ----------
  // every next word seen after `context`, backing off to shorter contexts when there is no match
  function countLookup(context) {
    const steps = [];
    for (let k = context.length; k >= 0; k--) {
      const shortened = context.slice(context.length - k);
      const next = new Array(V).fill(0);
      for (let i = 0; i + k < ids.length; i++) {
        if (shortened.every((word, j) => ids[i + j] === word)) next[ids[i + k]] += 1;
      }
      const total = next.reduce((a, b) => a + b, 0);
      steps.push({ context: shortened, next, total });
      if (total > 0) {
        let best = 0;
        next.forEach((count, word) => { if (count > next[best]) best = word; });
        // ties: Python keeps the first (context, word) pair it met in the text, i.e. the word seen first
        const firstSeen = firstOccurrence(shortened, next[best]);
        return { steps, used: k, next, total, prediction: firstSeen ?? best };
      }
    }
    throw new Error('The empty context always matches.');
  }

  function firstOccurrence(context, bestCount) {
    const k = context.length, seen = new Map();
    for (let i = 0; i + k < ids.length; i++) {
      if (context.every((word, j) => ids[i + j] === word)) {
        const word = ids[i + k];
        seen.set(word, (seen.get(word) || 0) + 1);
      }
    }
    // Python walks the Counter in insertion order and keeps the first word reaching the top count
    for (const [word, count] of seen) if (count === bestCount) return word;
    return null;
  }

  // ---------- neural model ----------
  function random(seed) {
    // mulberry32: a small repeatable random number generator
    let a = seed >>> 0;
    const uniform = () => {
      a = (a + 0x6D2B79F5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
    const normal = () => Math.sqrt(-2 * Math.log(1 - uniform())) * Math.cos(2 * Math.PI * uniform());
    return { uniform, normal };
  }

  function initModel(n, seed = 42) {
    const rng = random(seed), C = n - 1;
    return {
      n, rng, epoch: 0,
      E: Array.from({ length: V }, () => Array.from({ length: D }, () => rng.normal() * 0.1)),
      W: Array.from({ length: C * D }, () => Array.from({ length: V }, () => rng.normal() * 0.1)),
      b: new Array(V).fill(0),
    };
  }

  const clone = model => ({ ...model, E: model.E.map(r => r.slice()), W: model.W.map(r => r.slice()), b: model.b.slice() });

  function softmax(scores) {
    const max = Math.max(...scores);
    const exp = scores.map(s => Math.exp(s - max));
    const sum = exp.reduce((a, b) => a + b, 0);
    return exp.map(e => e / sum);
  }

  // every intermediate value for one window, so the page can show each of them
  function forward(model, context) {
    const pieces = context.map(word => model.E[word].slice());
    const x = pieces.flat();
    const scores = model.b.map((bias, v) => x.reduce((sum, xi, i) => sum + xi * model.W[i][v], 0) + bias);
    return { pieces, x, scores, probabilities: softmax(scores) };
  }

  // one batch: the same maths as train_step() in the Python script, returns the loss and every gradient
  function trainStep(model, batch, learningRate = LEARNING_RATE) {
    const B = batch.length, C = model.n - 1;
    const dW = model.W.map(row => row.map(() => 0)), db = new Array(V).fill(0);
    const dXs = [], details = [];
    let loss = 0;
    for (const { context, target } of batch) {
      const f = forward(model, context);
      loss -= Math.log(f.probabilities[target] + 1e-12) / B;
      const dScores = f.probabilities.map((p, v) => (p - (v === target ? 1 : 0)) / B);
      f.x.forEach((xi, i) => dScores.forEach((g, v) => { dW[i][v] += xi * g; }));
      dScores.forEach((g, v) => { db[v] += g; });
      const dX = f.x.map((_, i) => dScores.reduce((sum, g, v) => sum + g * model.W[i][v], 0));
      dXs.push(dX);
      details.push({ ...f, dScores, dX, context, target });
    }
    model.W.forEach((row, i) => row.forEach((_, v) => { row[v] -= learningRate * dW[i][v]; }));
    model.b.forEach((_, v) => { model.b[v] -= learningRate * db[v]; });
    batch.forEach(({ context }, k) => context.forEach((word, c) => {
      for (let d = 0; d < D; d++) model.E[word][d] -= learningRate * dXs[k][c * D + d];
    }));
    return { loss, dW, db, details };
  }

  function evaluate(model) {
    const all = windows(model.n);
    let correct = 0, loss = 0;
    for (const { context, target } of all) {
      const { probabilities } = forward(model, context);
      let best = 0;
      probabilities.forEach((p, v) => { if (p > probabilities[best]) best = v; });
      correct += best === target;
      loss -= Math.log(probabilities[target] + 1e-12);
    }
    return { accuracy: correct / all.length, loss: loss / all.length };
  }

  function trainEpoch(model) {
    const all = windows(model.n);
    // shuffle (Fisher-Yates) with the model's own random numbers
    for (let i = all.length - 1; i > 0; i--) {
      const j = Math.floor(model.rng.uniform() * (i + 1));
      [all[i], all[j]] = [all[j], all[i]];
    }
    let total = 0;
    for (let start = 0; start < all.length; start += BATCH_SIZE) {
      const batch = all.slice(start, start + BATCH_SIZE);
      total += trainStep(model, batch).loss * batch.length;
    }
    model.epoch += 1;
    return total / all.length;
  }

  function generateGreedy(predict, prompt, contextLength, words) {
    const text = prompt.slice();
    for (let i = 0; i < words; i++) text.push(predict(text.slice(-contextLength)));
    return text;
  }

  return { CORPUS, D, LEARNING_RATE, BATCH_SIZE, tokens, vocab, wordToId, ids, V, counts, windows,
    countLookup, initModel, clone, softmax, forward, trainStep, evaluate, trainEpoch, generateGreedy };
})();

if (typeof module !== 'undefined') module.exports = NgramEngine;
