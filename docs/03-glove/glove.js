const GloveReplay = (() => {
  const phases = [
    ['counts', 'Count contexts', 'Preparation'],
    ['log', 'Log targets', 'Preparation'],
    ['weight', 'Weight observations', 'Preparation'],
    ['lookup', 'Look up vectors', 'Forward pass'],
    ['predict', 'Predict log counts', 'Forward pass'],
    ['loss', 'Measure batch loss', 'Forward pass'],
    ['zero', 'Clear old gradients', 'Backward pass'],
    ['backward', 'Backpropagate', 'Backward pass'],
    ['adagrad', 'Adagrad update', 'Optimizer'],
    ['embedding', 'Inspect embeddings', 'Inspection'],
  ];
  const escape = value => String(value).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/"/g, '&quot;');
  const format = value => Math.abs(value) > 0 && Math.abs(value) < 0.0001
    ? value.toExponential(5) : String(Number(value.toFixed(6)));
  const dot = (a, b) => a.reduce((sum, v, i) => sum + v * b[i], 0);

  class Replay {
    constructor(metadata, buffer) {
      if (metadata.schemaVersion !== 1 || metadata.dtype !== 'float32' || metadata.byteOrder !== 'little' ||
          buffer.byteLength !== metadata.byteLength || buffer.byteLength % 4) {
        throw new Error('The recorded GloVe data has an unsupported format or incorrect byte length.');
      }
      this.m = metadata;
      const view = new DataView(buffer);
      this.floats = new Float32Array(buffer.byteLength / 4);
      for (let i = 0; i < this.floats.length; i++) this.floats[i] = view.getFloat32(i * 4, true);
      if (!this.floats.every(Number.isFinite)) throw new Error('The recording contains non-finite training values.');
      this.pairIndex = new Map(metadata.pairs.map(([i, j], k) => [`${i},${j}`, k]));
      this.layouts = new Map(metadata.layout.map(item => [item.name, item]));
      const parameterCount = metadata.layout.reduce((sum, item) => sum + item.length, 0);
      const required = { parameters: [metadata.config.epochs + 1, parameterCount],
        gradients: [metadata.config.epochs, parameterCount], accumulators: [metadata.config.epochs + 1, parameterCount],
        predictions: [metadata.config.epochs + 1, metadata.pairs.length], pair_losses: [metadata.config.epochs + 1, metadata.pairs.length] };
      let end = 0;
      for (const [name, [frames, width]] of Object.entries(required)) {
        const block = metadata.blocks[name];
        if (!block || block.frames !== frames || block.width !== width || block.offset !== end) {
          throw new Error(`Incorrect recorded tensor layout for ${name}. Regenerate the browser data.`);
        }
        end += frames * width;
      }
      if (end !== this.floats.length || metadata.losses.length !== metadata.config.epochs + 1 ||
          metadata.counts.length !== metadata.pairs.length || metadata.logCounts.length !== metadata.pairs.length ||
          metadata.weights.length !== metadata.pairs.length || !metadata.losses.every(Number.isFinite)) {
        throw new Error('Recorded losses, observations, and model tensors are inconsistent.');
      }
    }
    values(blockName, epoch) {
      const block = this.m.blocks[blockName];
      if (!block || !Number.isInteger(epoch) || epoch < 0 || epoch >= block.frames) {
        throw new RangeError(`No recorded ${blockName} tensor at checkpoint ${epoch}.`);
      }
      const start = block.offset + epoch * block.width;
      return this.floats.subarray(start, start + block.width);
    }
    row(name, epoch, word) {
      const layout = this.layouts.get(name);
      if (!layout || !Number.isInteger(word) || word < 0 || word >= layout.shape[0]) {
        throw new RangeError(`No ${name} row for word ${word}.`);
      }
      const width = layout.length / layout.shape[0], start = layout.offset + word * width;
      return this.values('parameters', epoch).subarray(start, start + width);
    }
    cell(epoch, pair) {
      if (!Number.isInteger(pair) || pair < 0 || pair >= this.m.pairs.length) throw new RangeError('Choose an observed pair.');
      const [i, j] = this.m.pairs[pair];
      const w = this.row('target_embeddings.weight', epoch, i), u = this.row('context_embeddings.weight', epoch, j);
      const bi = this.row('target_bias.weight', epoch, i)[0], bj = this.row('context_bias.weight', epoch, j)[0];
      const prediction = this.values('predictions', epoch)[pair], target = this.m.logCounts[pair];
      return { i, j, w, u, bi, bj, dot: dot(w, u), prediction, target, residual: prediction - target,
        count: this.m.counts[pair], weight: this.m.weights[pair], pairLoss: this.values('pair_losses', epoch)[pair] };
    }
    parameter(type, pair, coordinate) {
      const [i, j] = this.m.pairs[pair];
      const layout = this.layouts.get(type);
      if (!layout) throw new RangeError('Choose a target/context vector or bias.');
      const word = type.startsWith('target') ? i : j, width = layout.length / layout.shape[0];
      const dimension = width === 1 ? 0 : coordinate;
      if (!Number.isInteger(dimension) || dimension < 0 || dimension >= width) throw new RangeError('Invalid vector coordinate.');
      return { index: layout.offset + word * width + dimension, word, dimension, type,
        label: `${type.replace('.weight', '')}[${this.m.vocab[word]}${width > 1 ? `, ${dimension}` : ''}]` };
    }
    coefficients(epoch, parameter) {
      return this.m.pairs.map(([i, j]) => {
        if (parameter.type === 'target_embeddings.weight') return i === parameter.word ? this.row('context_embeddings.weight', epoch, j)[parameter.dimension] : 0;
        if (parameter.type === 'context_embeddings.weight') return j === parameter.word ? this.row('target_embeddings.weight', epoch, i)[parameter.dimension] : 0;
        return (parameter.type === 'target_bias.weight' ? i : j) === parameter.word ? 1 : 0;
      });
    }
    gradient(epoch, parameter) {
      const predictions = this.values('predictions', epoch), coefficients = this.coefficients(epoch, parameter);
      return coefficients.reduce((sum, coefficient, k) =>
        sum + 2 * this.m.weights[k] * (predictions[k] - this.m.logCounts[k]) * coefficient / this.m.pairs.length, 0);
    }
    lossAt(epoch, parameter, value) {
      const before = this.values('parameters', epoch)[parameter.index], coefficients = this.coefficients(epoch, parameter);
      const predictions = this.values('predictions', epoch), change = value - before;
      return coefficients.reduce((sum, coefficient, k) => {
        const residual = predictions[k] + change * coefficient - this.m.logCounts[k];
        return sum + this.m.weights[k] * residual * residual;
      }, 0) / this.m.pairs.length;
    }
    contributions(pair) {
      const [target, context] = this.m.pairs[pair], result = [];
      this.m.sentences.forEach((sentence, s) => sentence.forEach((word, i) => {
        if (word !== this.m.vocab[target]) return;
        sentence.forEach((candidate, j) => {
          const distance = Math.abs(i - j);
          if (candidate === this.m.vocab[context] && distance > 0 && distance <= this.m.config.window) {
            result.push({ sentence: s + 1, text: sentence.join(' '), distance, increment: 1 / distance });
          }
        });
      }));
      return result;
    }
    neighbors(epoch, word) {
      const embeddings = this.m.vocab.map((_, i) => {
        const a = this.row('target_embeddings.weight', epoch, i), b = this.row('context_embeddings.weight', epoch, i);
        return Array.from(a, (v, d) => Math.fround(v + b[d]));
      });
      const v = embeddings[word], norm = a => Math.sqrt(dot(a, a));
      return embeddings.map((u, i) => ({ word: this.m.vocab[i], id: i, cosine: dot(v, u) / (norm(v) * norm(u) + 1e-10) }))
        .filter(item => item.id !== word).sort((a, b) => b.cosine - a.cosine);
    }
  }

  async function load() {
    const response = await fetch('training.json', { cache: 'no-cache' });
    if (!response.ok) throw new Error(`Could not load GloVe metadata (HTTP ${response.status}).`);
    const metadata = await response.json();
    if (metadata.binaryFile !== 'training.f32') throw new Error('Unexpected GloVe recording filename.');
    const binaryResponse = await fetch('training.f32');
    if (!binaryResponse.ok) throw new Error(`Could not load GloVe tensors (HTTP ${binaryResponse.status}).`);
    const buffer = await binaryResponse.arrayBuffer();
    const hash = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', buffer)), b => b.toString(16).padStart(2, '0')).join('');
    if (hash !== metadata.binarySha256) throw new Error('The recording and metadata do not match. Reload or regenerate both files.');
    return new Replay(metadata, buffer);
  }

  function chart(series, { title, logarithmic = false, target = null } = {}) {
    const width = 640, height = 300, left = 82, right = 24, top = 28, bottom = 50;
    const all = series.map(p => p[1]);
    if (target !== null) all.push(target);
    if (!all.length || !all.every(Number.isFinite)) throw new Error('The chart needs finite recorded values.');
    const transform = value => logarithmic ? Math.log10(Math.max(value, 1e-20)) : value;
    let low = logarithmic ? Math.min(...all.map(transform)) : Math.min(0, ...all);
    let high = Math.max(...all.map(transform)), pad = Math.max((high - low) * 0.1, logarithmic ? 1 : 0.001);
    high += pad; low = logarithmic ? low - pad : Math.min(0, low - pad);
    const x = e => left + (width - left - right) * e / 300;
    const y = value => top + (height - top - bottom) * (high - transform(value)) / (high - low);
    let svg = `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${escape(title)}"><title>${escape(title)}</title><desc>Recorded PyTorch values through the selected checkpoint.${logarithmic ? ' The vertical axis is logarithmic; zero is displayed at 1e-20.' : ''}</desc>`;
    for (let i = 0; i <= 4; i++) {
      const value = low + (high - low) * i / 4, at = top + (height - top - bottom) * (4 - i) / 4;
      svg += `<line x1="${left}" x2="${width-right}" y1="${at}" y2="${at}" stroke="var(--line)"/>
        <text x="${left-8}" y="${at+5}" text-anchor="end">${logarithmic ? `10^${value.toFixed(1)}` : Number(value.toPrecision(3))}</text>`;
    }
    for (const e of [0, 50, 100, 150, 200, 250, 300]) svg += `<text x="${x(e)}" y="${height-24}" text-anchor="middle">${e}</text>`;
    if (target !== null) svg += `<line x1="${left}" x2="${width-right}" y1="${y(target)}" y2="${y(target)}" stroke="var(--target)" stroke-dasharray="6 5"/>`;
    const path = series.map(([e, value], i) => `${i ? 'L' : 'M'}${x(e)},${y(value)}`).join(' ');
    const last = series.at(-1);
    svg += `<path d="${path}" fill="none" stroke="var(--center)" stroke-width="2.5"/><circle cx="${x(last[0])}" cy="${y(last[1])}" r="4" fill="var(--center)"/>
      <text x="${left}" y="18">${logarithmic ? 'loss (log scale)' : 'recorded value'}</text><text x="${width/2}" y="${height-3}" text-anchor="middle">completed optimizer updates</text></svg>`;
    return svg;
  }
  return { Replay, load, phases, escape, format, chart };
})();

if (typeof document !== 'undefined') {
  (async () => {
    const $ = id => document.getElementById(id), api = GloveReplay, f = api.format, esc = api.escape;
    const math = (strings, ...values) => `\\(${String.raw(strings, ...values.map(value =>
      typeof value === 'number' ? f(value).replace(/e([+-]?\d+)/, '\\times 10^{$1}') : value))}\\)`;
    const view = document.body.dataset.view;
    if (!['math', 'intuition'].includes(view)) throw new Error('Choose a chapter-specific GloVe view.');
    if (!window.MathJax?.startup?.promise) throw new Error('Math typesetting could not load. Check access to the MathJax CDN and reload.');
    await MathJax.startup.promise;
    const replay = await api.load(), m = replay.m, phaseIds = api.phases.map(p => p[0]);
    let timer = null, storageBlocked = false;
    const key = 'glove-replay-v1';
    const state = { epoch: 0, pair: replay.pairIndex.get(`${m.vocab.indexOf('ice')},${m.vocab.indexOf('water')}`),
      coordinate: 0, parameter: 'target_bias.weight', phase: 'counts' };
    function notice(message) { $('storageNote').hidden = false; $('storageNote').textContent = message; }
    function valid(value) {
      return value && Number.isInteger(value.epoch) && value.epoch >= 0 && value.epoch <= m.config.epochs &&
        Number.isInteger(value.pair) && value.pair >= 0 && value.pair < m.pairs.length &&
        Number.isInteger(value.coordinate) && value.coordinate >= 0 && value.coordinate < m.config.dimensions &&
        replay.layouts.has(value.parameter) && phaseIds.includes(value.phase);
    }
    let saved = null;
    try { saved = localStorage.getItem(key); } catch (error) {
      console.warn('Could not read the saved GloVe replay position.', error);
      notice('Saved positions are unavailable. The lesson still works in this tab.'); storageBlocked = true;
    }
    if (saved) {
      try {
        const parsed = JSON.parse(saved);
        if (valid(parsed)) Object.assign(state, parsed);
        else { console.warn('Ignored invalid saved GloVe replay position.'); notice('The saved replay position was invalid; the lesson starts at checkpoint 0.'); }
      } catch (error) {
        console.warn('Ignored malformed saved GloVe replay position.', error);
        notice('The saved replay position was unreadable; the lesson starts at checkpoint 0.');
      }
    }
    const query = new URLSearchParams(location.search);
    if (['epoch', 'pair', 'phase'].some(name => query.has(name))) {
      const candidate = { ...state };
      for (const name of ['epoch', 'pair']) if (query.has(name)) candidate[name] = query.get(name).trim() === '' ? NaN : Number(query.get(name));
      if (query.has('phase')) candidate.phase = query.get('phase');
      if (valid(candidate)) Object.assign(state, candidate);
      else { console.warn('Ignored invalid GloVe replay URL.'); notice('The URL contained an invalid replay position; using the saved or starting position.'); }
    }
    function save() {
      const url = new URL(location.href);
      url.searchParams.set('epoch', state.epoch); url.searchParams.set('pair', state.pair);
      if (view === 'math') url.searchParams.set('phase', state.phase);
      history.replaceState(null, '', url);
      if (!storageBlocked) {
        try { localStorage.setItem(key, JSON.stringify(state)); }
        catch (error) { console.warn('Could not save the GloVe replay position.', error); notice('Position saving is unavailable; controls still work.'); storageBlocked = true; }
      }
    }
    function stop() { if (timer !== null) clearInterval(timer); timer = null; $('play').textContent = view === 'math' ? 'Play steps' : 'Play training'; $('play').setAttribute('aria-pressed', 'false'); }
    function setContexts() {
      const i = Number($('target').value);
      $('context').innerHTML = m.pairs.map(([a, b], pair) => a === i ? `<option value="${pair}">${esc(m.vocab[b])} (count ${m.counts[pair]})</option>` : '').join('');
      if (m.pairs[state.pair][0] === i) $('context').value = state.pair;
      else state.pair = Number($('context').value);
    }
    function table(headers, rows, caption) {
      return `<div class="scroll" tabindex="0" role="region" aria-label="${esc(caption)}"><table class="data-table"><caption>${esc(caption)}</caption>
        <thead><tr>${headers.map(h => `<th scope="col">${h}</th>`).join('')}</tr></thead><tbody>${rows.map(row =>
          `<tr>${row.map(value => `<td>${value}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;
    }
    function updateInfo() {
      const from = Math.min(state.epoch, m.config.epochs - 1), to = from + 1;
      const parameter = replay.parameter(state.parameter, state.pair, state.coordinate);
      const oldValue = replay.values('parameters', from)[parameter.index], nextValue = replay.values('parameters', to)[parameter.index];
      const gradient = replay.values('gradients', from)[parameter.index];
      const oldSum = replay.values('accumulators', from)[parameter.index], nextSum = replay.values('accumulators', to)[parameter.index];
      return { from, to, parameter, oldValue, nextValue, gradient, oldSum, nextSum };
    }
    function updateTable(info) {
      return table(['Quantity', 'Recorded PyTorch value'], [
        ['Parameter', `<code>${esc(info.parameter.label)}</code>`],
        ['Weight before', f(info.oldValue)], [math` \text{Full-batch gradient } g`, f(info.gradient)],
        ['Accumulated squared gradients before', f(info.oldSum)], [math`A_{\mathrm{new}} = A_{\mathrm{old}} + g^2`, f(info.nextSum)],
        [math`\sqrt{A_{\mathrm{new}}} + \varepsilon`, f(Math.sqrt(info.nextSum) + m.optimizer.eps)],
        ['New weight', f(info.nextValue)], ['Actual change', f(info.nextValue - info.oldValue)],
      ], `Actual optimizer transition ${info.from} → ${info.to}`);
    }
    function contributionTable() {
      const [i, j] = m.pairs[state.pair];
      return table(['Sentence', 'Token distance', math`\text{Contribution to } X`], replay.contributions(state.pair).map(c =>
        [esc(c.text), c.distance, math`\frac{1}{${c.distance}} = ${c.increment}`]), `Evidence for ${m.vocab[i]} / ${m.vocab[j]}: count ${m.counts[state.pair]}`);
    }
    function weightGraph(cell) {
      const { xMax, alpha } = m.config;
      return MathGraphs.slope({ title: 'Counts determine how strongly a cell is weighted',
        context: `This curve shows confidence weighting, not the earlier inverse-distance context contribution. All observed counts here are 0.5, 1, or 2, below the count threshold of ${xMax}. The cap at 1 is not reached by this corpus.`,
        x: cell.count, xMin: 0, xMax: 20, xLabel: 'weighted co-occurrence count', yLabel: 'confidence weight',
        curves: [{ label: 'Confidence weight', fn: x => x < xMax ? (x / xMax) ** alpha : 1,
          slope: cell.count < xMax ? alpha * (cell.count / xMax) ** (alpha - 1) / xMax : 0 }] });
    }
    function errorGraph(cell) {
      return MathGraphs.slope({ title: 'Moving the prediction toward the log-count target reduces this cell’s error',
        context: 'The horizontal value is prediction minus target, not a probability. This curve divides by 80 because the batch loss averages all observed cells.',
        x: cell.residual, xMin: Math.min(-1, cell.residual - 0.5), xMax: Math.max(1, cell.residual + 0.5),
        xLabel: 'log-count residual', yLabel: 'contribution to mean loss',
        curves: [{ label: 'Weighted squared residual / 80', fn: r => cell.weight * r * r / m.pairs.length,
          slope: 2 * cell.weight * cell.residual / m.pairs.length }] });
    }
    function gradientTable(epoch, parameter) {
      const coefficients = replay.coefficients(epoch, parameter), predictions = replay.values('predictions', epoch);
      return table(['Observed cell', 'Residual', math`\frac{\partial y_{ij}}{\partial \theta}`, 'Contribution to batch gradient'],
        m.pairs.flatMap(([i, j], k) => coefficients[k] === 0 ? [] : [[`${esc(m.vocab[i])} / ${esc(m.vocab[j])}`,
          f(predictions[k] - m.logCounts[k]), f(coefficients[k]),
          f(2 * m.weights[k] * (predictions[k] - m.logCounts[k]) * coefficients[k] / m.pairs.length)]]),
        'Shared parameters collect contributions from every affected cell');
    }
    function sensitivityGraph(info) {
      return MathGraphs.slope({ title: 'One parameter, all affected errors',
        context: `At checkpoint ${info.from}, vary only ${info.parameter.label}; hold every other weight and bias fixed. This is the same full-batch objective as the probes, not a prediction of the next whole-model update.`,
        x: info.oldValue, xMin: info.oldValue - 0.05, xMax: info.oldValue + 0.05,
        xLabel: info.parameter.label, yLabel: 'full-batch mean loss',
        curves: [{ label: 'Loss with one parameter varied', fn: value => replay.lossAt(info.from, info.parameter, value),
          slope: replay.gradient(info.from, info.parameter) }] });
    }
    function renderMath(cell) {
      const phase = api.phases.find(p => p[0] === state.phase), info = updateInfo(), uCell = replay.cell(info.from, state.pair);
      const pair = `${esc(m.vocab[cell.i])} / ${esc(m.vocab[cell.j])}`;
      $('phaseTitle').textContent = `${phase[2]}: ${phase[1]}`;
      let why, worked, graph = '';
      const review = state.epoch === m.config.epochs ? `<p class="recording-note">All 300 updates are complete. This optimizer/backward inspection reviews the last actual transition, 299 → 300; there is no recorded update 301.</p>` : '';
      if (state.phase === 'counts') {
        why = 'Aggregate repeated context evidence once before training. Close words contribute more than distant words; sentence boundaries stop the window. A unique observed cell is one batch item, even if many token occurrences contributed to it.';
        worked = contributionTable() + `<p class="code-equation">For ${pair}: ${math`X_{ij} = ${cell.count}`}. The reverse cell is counted separately. Only the 80 nonzero directed cells enter the objective.</p>`;
      } else if (state.phase === 'log') {
        why = 'Fit a compressed count scale. Equal count ratios become equal log differences, so frequent observations do not require proportionally enormous dot products. A count below one gives a valid negative target.';
        worked = `<p class="code-equation">${math`t_{ij} = \log(${cell.count}) = ${cell.target}`}</p><p>Log means base ${math`e`}. A zero cell has no log target and is omitted, not replaced by ${math`\log(0 + \varepsilon)`}.</p>`;
      } else if (state.phase === 'weight') {
        why = `Counts are noisy evidence. The heuristic ${math`f(X)`} gives a rarer cell less influence at the same residual and caps the multiplier on very frequent cells. It is separate from the inverse-distance weighting that built ${math`X`}.`;
        worked = `<p class="code-equation">${math`f(${cell.count}) = \left(\frac{${cell.count}}{10}\right)^{0.75} = ${cell.weight}`}</p><p>The target remains ${math`\log(X)`}; ${math`f(X)`} scales the cost rather than changing the target.</p>`;
        graph = weightGraph(cell);
      } else if (state.phase === 'lookup') {
        why = 'Select the vectors and biases for each observed cell. Target and context roles have separate parameters. The selected cell is a lens on a full 80-cell forward pass, not a single-pair training update.';
        worked = table(['Parameter', 'Values'], [
          [`Target vector: ${esc(m.vocab[cell.i])}`, `<code>${Array.from(cell.w, f).join(', ')}</code>`],
          [`Context vector: ${esc(m.vocab[cell.j])}`, `<code>${Array.from(cell.u, f).join(', ')}</code>`],
          ['Target bias', f(cell.bi)], ['Context bias', f(cell.bj)],
        ], 'The selected rows and scalar biases');
      } else if (state.phase === 'predict') {
        why = 'Combine the two word roles into an unrestricted log-count prediction. Biases allow word-specific offsets so the interaction vectors need not carry every baseline frequency effect. There is no sigmoid or softmax.';
        worked = table(['Coordinate', 'Target value', 'Context value', 'Product'],
          Array.from(cell.w, (v, d) => [d, f(v), f(cell.u[d]), f(v * cell.u[d])]), 'Dot product across all 20 coordinates') +
          `<p class="code-equation">${math`y_{ij} = w_i^\top u_j + b_i + c_j`}<br>${math`${cell.dot} + (${cell.bi}) + (${cell.bj}) \approx ${cell.prediction}`}</p>
          <p>Recorded prediction ${math`y_{ij} = ${cell.prediction}`}. ${math`\exp(y_{ij}) \approx ${Math.exp(cell.prediction)}`} is a count-scale reconstruction, not a probability. Displayed decompositions use JavaScript arithmetic on the recorded float32 weights; tiny rounding differences are normal.</p>`;
      } else if (state.phase === 'loss') {
        why = `A prediction needs a standard for improvement. Squaring the difference from ${math`\log(X)`} penalizes errors in either direction; ${math`f(X)`} sets the relative influence, and <code>mean()</code> averages the 80 unique observed cells.`;
        worked = `<p class="code-equation">${math`r_{ij} = ${cell.prediction} - (${cell.target}) = ${cell.residual}`}<br>
          ${math`f(X_{ij})r_{ij}^2 = ${cell.weight} \times r_{ij}^2 = ${cell.pairLoss}`}<br>${math`\frac{f(X_{ij})r_{ij}^2}{M} = ${cell.pairLoss / m.pairs.length}`}</p>
          <p>Recorded full-batch ${math`L = ${m.losses[state.epoch]}`}. It includes all cells, not just ${pair}.</p>`;
        graph = errorGraph(cell);
      } else if (state.phase === 'zero') {
        why = 'PyTorch adds new gradients to existing .grad fields by default. Clear the old batch gradients so backward() supplies only the current objective’s gradient. This does not clear Adagrad’s running squared-gradient history.';
        worked = review + '<p class="code-equation"><code>optimizer.zero_grad()</code></p><p>Parameters are unchanged. The next backward pass fills their gradients; Adagrad will use its separate persistent accumulator when step() is called.</p>';
      } else if (state.phase === 'backward') {
        why = 'Autograd follows the graph built by embedding lookup, multiplication, sums, biases, squaring, weighting, and mean. It accumulates contributions whenever an embedding row or bias was reused by several cells.';
        worked = review + `<p>Backward values below refer to checkpoint ${info.from}, before optimizer update ${info.to}.</p>
          <p class="code-equation">${math`\frac{\partial L}{\partial y_{ij}} = \frac{2f(X_{ij})r_{ij}}{80} = ${2 * uCell.weight * uCell.residual / m.pairs.length}`}</p>` +
          gradientTable(info.from, info.parameter) +
          `<p>Sum from the displayed smooth formula ≈ ${f(replay.gradient(info.from, info.parameter))}. Recorded PyTorch .grad = ${f(info.gradient)}. The latter is the value passed to Adagrad.</p>`;
        graph = sensitivityGraph(info);
      } else if (state.phase === 'adagrad') {
        why = 'Use the full-batch gradient to change the parameters once. Adagrad keeps a separate sum of squared past gradients for every scalar parameter, scaling its step by the square root of that history.';
        worked = review + `<p class="code-equation">${math`A_{\mathrm{new}} = A_{\mathrm{old}} + g^2`}<br>${math`\theta_{\mathrm{new}} = \theta_{\mathrm{old}} - \frac{0.03g}{\sqrt{A_{\mathrm{new}}}+10^{-10}}`}</p>` +
          updateTable(info) + `<p>The accumulator starts at zero and is not reset by zero_grad(). Defaults here use no weight decay and no learning-rate decay. No shuffle or per-cell optimizer step is present.</p>`;
        graph = MathGraphs.slope({ title: 'Local batch loss around the inspected parameter',
          context: `Only ${info.parameter.label} varies; all other parameters are fixed at checkpoint ${info.from}. The curve evaluates the smooth objective from recorded float32 values. Its slope agrees with autograd up to floating-point precision.`,
          x: info.oldValue, xMin: info.oldValue - 0.2, xMax: info.oldValue + 0.2,
          xLabel: 'parameter value', yLabel: 'mean batch loss',
          curves: [{ label: 'All 80 cells', fn: value => replay.lossAt(info.from, info.parameter, value),
            slope: replay.gradient(info.from, info.parameter) }] });
      } else {
        why = 'After training, this implementation adds each word’s target and context vectors for similarity queries. This combines the two learned roles but is not the formula used to predict a cell.';
        worked = `<p class="code-equation">${math`e_i = w_i + u_i`}</p><p>The script extracts these after training. Inspecting them at earlier checkpoints is a visualization, not another operation inside the optimizer loop.</p>` + embeddingTable(cell);
      }
      $('whyStep').innerHTML = why; $('workedStep').innerHTML = worked; $('stepGraph').innerHTML = graph;
      document.querySelectorAll('#phases button').forEach(button => button.setAttribute('aria-pressed', String(button.dataset.phase === state.phase)));
      const activeLines = m.phaseLines[state.phase];
      document.querySelectorAll('.source-line').forEach(line => line.classList.toggle('active', activeLines.includes(Number(line.dataset.line))));
      const line = $('codeSource').querySelector('.active');
      if (line) $('codeSource').scrollTop = Math.max(0, line.offsetTop - $('codeSource').offsetTop - 60);
    }
    function embeddingTable(cell) {
      return table(['Coordinate', 'Target role', 'Context role of same word', 'Sum'], Array.from(cell.w, (v, d) => {
        const context = replay.row('context_embeddings.weight', state.epoch, cell.i)[d];
        return [d, f(v), f(context), f(Math.fround(v + context))];
      }), `Combined embedding of ${m.vocab[cell.i]}`);
    }
    function renderIntuition(cell) {
      const info = updateInfo(), basis = replay.cell(info.from, state.pair);
      $('countContributions').innerHTML = contributionTable();
      $('logExample').innerHTML = `<p class="code-equation">For ${esc(m.vocab[cell.i])} / ${esc(m.vocab[cell.j])}: ${math`X_{ij} = ${cell.count}`}, so the target is ${math`\log(X_{ij}) = ${cell.target}`}.</p>`;
      $('weightExample').innerHTML = `<p class="code-equation">This cell’s confidence multiplier is ${math`\left(\frac{${cell.count}}{10}\right)^{0.75} = ${cell.weight}`}.</p>`;
      $('weightGraph').innerHTML = weightGraph(cell);
      $('predictionExample').innerHTML = `<p>For ${esc(m.vocab[cell.i])} / ${esc(m.vocab[cell.j])} at checkpoint ${state.epoch}, the 20-coordinate dot product is approximately ${f(cell.dot)}. Its target bias is ${f(cell.bi)} and context bias is ${f(cell.bj)}.</p>
        <p class="code-equation">${math`y_{ij} = w_i^\top u_j + b_i + c_j = ${cell.prediction}`} (recorded PyTorch prediction)</p>
        <p>The observed target is ${f(cell.target)}. The reconstructed count ${math`\exp(y_{ij}) \approx ${Math.exp(cell.prediction)}`}, compared with observed ${math`X_{ij} = ${cell.count}`}.</p>`;
      $('lossExample').innerHTML = `<p class="code-equation">${math`r_{ij} = ${cell.prediction} - (${cell.target}) = ${cell.residual}`}<br>
        ${math`f(X_{ij})r_{ij}^2 = ${cell.pairLoss}`}<br>${math`\frac{f(X_{ij})r_{ij}^2}{80} = ${cell.pairLoss / m.pairs.length}`}</p>`;
      $('errorGraph').innerHTML = errorGraph(cell);
      $('updateNote').textContent = state.epoch === 300 ? 'The run is complete. The following gradient and optimizer inspection reviews its last actual update, 299 → 300.' : `The next actual update goes from checkpoint ${info.from} to ${info.to}.`;
      $('gradientExample').innerHTML = `<p>For the selected cell at checkpoint ${info.from}, ${math`\frac{\partial L}{\partial y_{ij}} = ${2 * basis.weight * basis.residual / m.pairs.length}`}.
        The optimizer’s recorded gradient for <code>${esc(info.parameter.label)}</code> is <code>${f(info.gradient)}</code>, after adding all affected cells’ contributions.</p>` + gradientTable(info.from, info.parameter);
      const probes = [-0.01, 0, 0.01].map(delta => [delta < 0 ? 'Lower by 0.01' : delta > 0 ? 'Raise by 0.01' : 'Leave unchanged',
        f(info.oldValue + delta), f(replay.lossAt(info.from, info.parameter, info.oldValue + delta))]);
      $('sensitivityTable').innerHTML = table(['Probe', 'Parameter value', 'Full-batch loss'], probes, 'Hold every other parameter fixed');
      $('sensitivityGraph').innerHTML = sensitivityGraph(info);
      $('adagradExample').innerHTML = updateTable(info);
      $('combinedEmbedding').innerHTML = embeddingTable(cell);
    }
    function renderMatrix(cell) {
      $('matrix').innerHTML = `<table class="cooccurrence"><caption>Observed distance-weighted counts X; not model predictions</caption><thead><tr><th></th>${m.vocab.map(w => `<th scope="col">${esc(w)}</th>`).join('')}</tr></thead><tbody>
        ${m.vocab.map((word, i) => `<tr><th scope="row" class="word-label">${esc(word)}</th>${m.vocab.map((context, j) =>
          `<td><button type="button" data-cell="${i},${j}" class="${m.matrix[i][j] > 0 ? 'observed' : ''}${i === cell.i && j === cell.j ? ' selected' : ''}" aria-label="${esc(word)} / ${esc(context)}: count ${m.matrix[i][j]}">${m.matrix[i][j]}</button></td>`).join('')}</tr>`).join('')}</tbody></table>`;
    }
    const mathRegions = [
      'whyStep', 'workedStep', 'countContributions', 'logExample', 'weightExample', 'predictionExample',
      'lossExample', 'gradientExample', 'adagradExample', 'recording', 'fitCaption',
    ].map($).filter(Boolean);
    let rendering = Promise.resolve();
    function render() {
      // Serialize DOM replacement and typesetting so playback cannot race MathJax.
      rendering = rendering.then(async () => {
        MathJax.typesetClear(mathRegions);
        renderContent();
        await MathJax.typesetPromise(mathRegions);
      }).catch(error => {
        stop();
        console.error('GloVe math rendering failed.', error);
        $('loadError').hidden = false;
        $('loadError').textContent = `Could not render the lesson mathematics: ${error.message}. Reload the page to retry.`;
      });
      return rendering;
    }
    function renderContent() {
      $('recording').innerHTML = `Recorded with PyTorch ${esc(m.torchVersion)}, CPU float32. ${math`D = 20`}, window of 2, 300 full-batch updates, Adagrad ${math`\eta = 0.03`}, ${math`X_{\max} = 10`}, ${math`\alpha = 0.75`}, seed 42.`;
      const cell = replay.cell(state.epoch, state.pair);
      $('target').value = cell.i; $('context').value = state.pair;
      $('coordinate').value = state.coordinate; $('parameter').value = state.parameter;
      $('coordinate').disabled = state.parameter.includes('_bias');
      $('epoch').value = state.epoch; $('epochReadout').textContent = `${state.epoch} / ${m.config.epochs}`;
      $('status').textContent = `Checkpoint ${state.epoch}: ${state.epoch} full-batch optimizer updates completed. Inspecting ${m.vocab[cell.i]} / ${m.vocab[cell.j]}. Selecting another cell does not update the model.`;
      $('metrics').innerHTML = [
        ['Observed X', cell.count], ['Log-count target', f(cell.target)], ['Recorded prediction', f(cell.prediction)],
        ['Full-batch mean loss', f(m.losses[state.epoch])],
      ].map(([label, value]) => `<p>${label}<strong>${value}</strong></p>`).join('');
      const lossSeries = m.losses.slice(0, state.epoch + 1).map((value, e) => [e, value]);
      $('lossPlot').innerHTML = api.chart(lossSeries, { title: 'Actual full-batch objective', logarithmic: $('logScale').checked });
      $('fitPlot').innerHTML = api.chart(Array.from({ length: state.epoch + 1 }, (_, e) => [e, replay.values('predictions', e)[state.pair]]),
        { title: 'Selected log-count prediction versus fixed observed target', target: cell.target });
      $('fitCaption').innerHTML = `Teal: predicted log count for ${esc(m.vocab[cell.i])} / ${esc(m.vocab[cell.j])}. Ochre dashed: its fixed target ${math`\log(X_{ij}) = ${cell.target}`}.`;
      $('lossCaption').textContent = $('logScale').checked
        ? 'A log vertical axis exposes late changes hidden near zero on a linear scale. Equal vertical distances mean equal ratios; any exact zero is plotted at 1e-20, without changing the recorded loss.'
        : 'Loss evaluated at each recorded checkpoint. The reference script logs just before each update: its “epoch 50” line is checkpoint 49, not 50.';
      $('neighbors').innerHTML = table(['Closest combined embeddings', 'Cosine'], replay.neighbors(state.epoch, cell.i).slice(0, 5).map(n => [esc(n.word), f(n.cosine)]), `Actual nearest neighbours of ${m.vocab[cell.i]} at this checkpoint`);
      if (view === 'math') { renderMath(cell); renderMatrix(cell); } else renderIntuition(cell);
      $('previous').disabled = state.epoch === 0 && (view === 'intuition' || state.phase === 'counts');
      $('next').disabled = state.epoch === 300 && (view === 'intuition' || state.phase === 'embedding');
      $('end').disabled = state.epoch === 300;
      if ($('nextEpoch')) $('nextEpoch').disabled = state.epoch === 300;
      save();
    }
    function move(direction) {
      if (view === 'intuition') {
        const next = Math.max(0, Math.min(300, state.epoch + direction));
        if (next === state.epoch) return false;
        state.epoch = next;
      } else {
        const index = phaseIds.indexOf(state.phase), next = index + direction;
        if (direction < 0 && state.phase === 'lookup' && state.epoch > 0) { state.epoch--; state.phase = 'embedding'; }
        else if (next < 0) { if (state.epoch === 0) return false; state.epoch--; state.phase = 'embedding'; }
        else if (next === phaseIds.length) { if (state.epoch === 300) return false; state.epoch++; state.phase = 'lookup'; }
        else state.phase = phaseIds[next];
      }
      render(); return true;
    }
    $('target').innerHTML = m.vocab.map((word, i) => `<option value="${i}">${esc(word)}</option>`).join('');
    $('target').value = m.pairs[state.pair][0]; setContexts();
    $('coordinate').innerHTML = Array.from({ length: m.config.dimensions }, (_, i) => `<option value="${i}">${i}</option>`).join('');
    $('corpus').textContent = m.corpus.join('\n');
    $('target').onchange = () => { stop(); setContexts(); render(); };
    $('context').onchange = () => { stop(); state.pair = Number($('context').value); render(); };
    $('coordinate').onchange = () => { state.coordinate = Number($('coordinate').value); render(); };
    $('parameter').onchange = () => { state.parameter = $('parameter').value; render(); };
    $('epoch').oninput = () => { stop(); state.epoch = Number($('epoch').value); render(); };
    $('logScale').onchange = render;
    $('start').onclick = () => { stop(); state.epoch = 0; state.phase = 'counts'; render(); };
    $('end').onclick = () => { stop(); state.epoch = 300; render(); };
    $('previous').onclick = () => { stop(); move(-1); };
    $('next').onclick = () => { stop(); move(1); };
    if ($('nextEpoch')) $('nextEpoch').onclick = () => { stop(); state.epoch = Math.min(300, state.epoch + 1); state.phase = 'lookup'; render(); };
    $('play').onclick = () => {
      if (timer !== null) { stop(); return; }
      $('play').textContent = 'Pause'; $('play').setAttribute('aria-pressed', 'true');
      timer = setInterval(() => { if (!move(1)) stop(); }, view === 'math' ? 700 : 160);
    };
    if (view === 'math') {
      $('phases').innerHTML = api.phases.map(([phase, label, kind]) => `<button type="button" data-phase="${phase}" title="${kind}" aria-pressed="false">${label}</button>`).join('');
      $('phases').onclick = event => { const button = event.target.closest('[data-phase]'); if (button) { stop(); state.phase = button.dataset.phase; render(); } };
      const linePhases = new Map();
      for (const [phase, lines] of Object.entries(m.phaseLines)) for (const line of lines) linePhases.set(line, phase);
      $('codeSource').innerHTML = m.source.split('\n').map((text, index) => {
        const phase = linePhases.get(index + 1);
        return `<${phase ? 'button type="button"' : 'div'} class="source-line" data-line="${index+1}"${phase ? ` data-phase="${phase}" title="Inspect ${phase}"` : ''}><span class="line-number">${index+1}</span><span>${esc(text)}</span></${phase ? 'button' : 'div'}>`;
      }).join('');
      $('codeSource').onclick = event => { const line = event.target.closest('[data-phase]'); if (line) { stop(); state.phase = line.dataset.phase; render(); } };
      $('matrix').onclick = event => {
        const button = event.target.closest('[data-cell]');
        if (!button) return;
        const pair = replay.pairIndex.get(button.dataset.cell);
        if (pair === undefined) {
          $('matrixNote').textContent = 'This cell is zero: no observed count, no log target, and no training loss term. GloVe does not treat it as a negative example. The previously selected observed cell remains selected.';
        } else {
          stop(); state.pair = pair; $('target').value = m.pairs[pair][0]; setContexts();
          $('matrixNote').textContent = 'Selected an observed cell. The model checkpoint is unchanged.'; render();
        }
      };
    }
    document.addEventListener('keydown', event => {
      if (event.target.closest('input,select,textarea,button,a,[tabindex]')) return;
      if (event.key === 'ArrowRight' || event.key === 'ArrowLeft') { event.preventDefault(); stop(); move(event.key === 'ArrowRight' ? 1 : -1); }
      else if (event.key === ' ') { event.preventDefault(); $('play').click(); }
    });
    window.addEventListener('pagehide', stop);
    $('lessonControls').disabled = false; $('loading').hidden = true;
    await render();
  })().catch(error => {
    console.error('GloVe replay failed.', error);
    document.getElementById('loading').hidden = true;
    const alert = document.getElementById('loadError');
    alert.hidden = false; alert.textContent = `${error.message} Serve the docs folder over HTTP; see the chapter README for setup.`;
    document.getElementById('lessonControls').disabled = true;
  });
}
