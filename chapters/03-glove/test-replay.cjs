const assert = require('node:assert/strict');
const { createHash } = require('node:crypto');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const root = path.resolve(__dirname, '../../docs/03-glove');
const metadata = JSON.parse(fs.readFileSync(path.join(root, 'training.json'), 'utf8'));
const binary = fs.readFileSync(path.join(root, 'training.f32'));
const context = vm.createContext({});
vm.runInContext(fs.readFileSync(path.join(root, 'glove.js'), 'utf8') + '\nglobalThis.api = GloveReplay;', context);
const buffer = binary.buffer.slice(binary.byteOffset, binary.byteOffset + binary.byteLength);
const replay = new context.api.Replay(metadata, buffer);
const types = metadata.layout.map(layout => layout.name);
const near = (actual, expected, tolerance, label) =>
  assert.ok(Math.abs(actual - expected) <= tolerance, `${label}: ${actual} vs ${expected}`);

test('recording and embedded source match their hashes and current Python source', () => {
  const sha = bytes => createHash('sha256').update(bytes).digest('hex');
  assert.equal(sha(binary), metadata.binarySha256);
  assert.equal(sha(metadata.source), metadata.sourceSha256);
  assert.equal(metadata.source, fs.readFileSync(path.join(__dirname, 'glove.py'), 'utf8'));
  for (const lines of Object.values(metadata.phaseLines)) {
    assert.ok(lines.length > 0 && lines.every(line => line >= 1 && line <= metadata.source.split('\n').length));
  }
});

test('all 301 states reconstruct recorded predictions, cell costs, and mean losses', () => {
  for (let epoch = 0; epoch <= metadata.config.epochs; epoch++) {
    let sum = 0;
    for (let pair = 0; pair < metadata.pairs.length; pair++) {
      const cell = replay.cell(epoch, pair);
      // PyTorch reduces float32 products; this explanatory dot product uses JavaScript doubles.
      near(cell.dot + cell.bi + cell.bj, cell.prediction, 2e-7, `prediction ${epoch}/${pair}`);
      near(cell.weight * cell.residual ** 2, cell.pairLoss, 2e-8, `cell loss ${epoch}/${pair}`);
      sum += cell.pairLoss;
    }
    near(sum / metadata.pairs.length, metadata.losses[epoch], 1e-8, `mean loss ${epoch}`);
  }
});

test('every full-batch scalar gradient matches the recorded PyTorch autograd result', () => {
  for (let epoch = 0; epoch < metadata.config.epochs; epoch++) {
    const recorded = replay.values('gradients', epoch);
    for (const layout of metadata.layout) {
      const width = layout.length / layout.shape[0];
      for (let word = 0; word < layout.shape[0]; word++) {
        for (let dimension = 0; dimension < width; dimension++) {
          const parameter = { type: layout.name, word, dimension };
          const index = layout.offset + word * width + dimension;
          near(replay.gradient(epoch, parameter), recorded[index], 1e-8, `gradient ${epoch}/${index}`);
        }
      }
    }
  }
});

test('all 264600 Adagrad transitions use the actual gradient and per-scalar history', () => {
  for (let epoch = 0; epoch < metadata.config.epochs; epoch++) {
    const before = replay.values('parameters', epoch), after = replay.values('parameters', epoch + 1);
    const oldSum = replay.values('accumulators', epoch), newSum = replay.values('accumulators', epoch + 1);
    const gradient = replay.values('gradients', epoch);
    for (let index = 0; index < before.length; index++) {
      near(newSum[index], oldSum[index] + gradient[index] ** 2, 2e-9, `accumulator ${epoch}/${index}`);
      const expected = before[index] - metadata.optimizer.lr * gradient[index] /
        (Math.sqrt(newSum[index]) + metadata.optimizer.eps);
      near(after[index], expected, 5e-8, `update ${epoch}/${index}`);
    }
  }
});

test('all cell selections bind role-specific parameters and explain their observed counts', () => {
  for (let pair = 0; pair < metadata.pairs.length; pair++) {
    const [i, j] = metadata.pairs[pair];
    near(replay.contributions(pair).reduce((sum, item) => sum + item.increment, 0), metadata.counts[pair], 1e-7, `count ${pair}`);
    for (const type of types) {
      const layout = replay.layouts.get(type), width = layout.length / layout.shape[0];
      for (const coordinate of [0, metadata.config.dimensions - 1]) {
        const parameter = replay.parameter(type, pair, coordinate);
        assert.equal(parameter.word, type.startsWith('target') ? i : j);
        assert.equal(parameter.index, layout.offset + parameter.word * width + (width === 1 ? 0 : coordinate));
        near(replay.row(type, 0, parameter.word)[parameter.dimension],
          replay.values('parameters', 0)[parameter.index], 0, 'lookup');
      }
    }
  }
});

test('one-parameter probes agree with analytical sensitivities', () => {
  for (const epoch of [0, 1, 10, 50, 299]) {
    for (const type of types) {
      const parameter = replay.parameter(type, 37, 19), value = replay.values('parameters', epoch)[parameter.index];
      const step = 1e-5;
      const derivative = (replay.lossAt(epoch, parameter, value + step) - replay.lossAt(epoch, parameter, value - step)) / (2 * step);
      near(derivative, replay.gradient(epoch, parameter), 1e-10, `sensitivity ${epoch}/${type}`);
    }
  }
});

test('combined-embedding rankings preserve the actual default-run counterexample', () => {
  const epoch = metadata.config.epochs, ice = metadata.vocab.indexOf('ice'), cats = metadata.vocab.indexOf('cats');
  const iceNeighbors = replay.neighbors(epoch, ice), catNeighbors = replay.neighbors(epoch, cats);
  assert.equal(iceNeighbors[0].word, 'into');
  assert.equal(catNeighbors[0].word, 'dogs');
  near(iceNeighbors.find(item => item.word === 'water').cosine, -0.7507, 0.00005, 'ice/water cosine');
  near(catNeighbors[0].cosine, 0.7455, 0.00005, 'cats/dogs cosine');
});

test('invalid checkpoints, pairs, coordinates, and corrupted layouts fail explicitly', () => {
  assert.throws(() => replay.values('gradients', 300), /No recorded/);
  assert.throws(() => replay.cell(0, 80), /observed pair/);
  assert.throws(() => replay.parameter(types[0], 37, 20), /coordinate/);
  assert.throws(() => replay.row(types[0], 0, -1), /row/);
  assert.throws(() => new context.api.Replay({ ...metadata, byteLength: 0 }, buffer), /byte length/);
  const corrupted = { ...metadata, blocks: { ...metadata.blocks, gradients: { ...metadata.blocks.gradients, frames: 1 } } };
  assert.throws(() => new context.api.Replay(corrupted, buffer), /tensor layout/);
});
