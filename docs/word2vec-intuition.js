const Word2VecIntuition = (() => {
  const corpus = [
    'cats furry pets', 'cats playful pets', 'cats gentle pets',
    'dogs furry pets', 'dogs playful pets', 'dogs gentle pets',
    'rabbits furry pets', 'rabbits playful pets', 'rabbits gentle pets',
    'apples sweet fruit', 'apples juicy fruit', 'apples fresh fruit',
    'bananas sweet fruit', 'bananas juicy fruit', 'bananas fresh fruit',
    'oranges sweet fruit', 'oranges juicy fruit', 'oranges fresh fruit',
    'cars fast vehicles', 'cars motorized vehicles', 'cars wheeled vehicles',
    'buses fast vehicles', 'buses motorized vehicles', 'buses wheeled vehicles',
    'trucks fast vehicles', 'trucks motorized vehicles', 'trucks wheeled vehicles'
  ];
  const words = [...new Set(corpus.flatMap(s => s.split(' ')))].sort();
  const pairs = [];
  const counts = words.map(() => 0);
  for (const sentence of corpus) {
    const ids = sentence.split(' ').map(w => words.indexOf(w));
    for (let i = 0; i < ids.length; i++) {
      counts[ids[i]]++;
      for (let j = Math.max(0, i - 1); j <= Math.min(ids.length - 1, i + 1); j++) {
        if (i !== j) pairs.push([ids[i], ids[j]]);
      }
    }
  }
  const powers = counts.map(n => n ** 0.75);
  const sum = powers.reduce((a, b) => a + b, 0);
  let cumulative = 0;
  const rawCDF = powers.map(n => cumulative += n / sum);
  const cdf = rawCDF.map(n => n / rawCDF.at(-1));
  const dimensions = 3, negatives = 3, maxEpochs = 200;
  const copy = matrix => matrix.map(row => row.slice());
  const dot = (a, b) => a.reduce((total, v, i) => total + v * b[i], 0);
  const sigmoid = s => 1 / (1 + Math.exp(-Math.max(-15, Math.min(15, s))));
  const softplus = s => s > 0 ? s + Math.log1p(Math.exp(-s)) : Math.log1p(Math.exp(s));
  function softmax(scores) {
    const max = Math.max(...scores), e = scores.map(s => Math.exp(s - max));
    const total = e.reduce((a, b) => a + b, 0);
    return e.map(v => v / total);
  }

  // Match the chapters' NumPy legacy RandomState, including Gaussian caching and shuffle.
  class NpRandom {
    constructor(seed) {
      this.mt = new Uint32Array(624); this.mt[0] = seed >>> 0;
      for (let i = 1; i < 624; i++) {
        const p = this.mt[i - 1] ^ (this.mt[i - 1] >>> 30);
        this.mt[i] = (Math.imul(1812433253, p) + i) >>> 0;
      }
      this.i = 624; this.hasG = false;
    }
    gen() {
      const m = this.mt;
      for (let k = 0; k < 624; k++) {
        const y = (m[k] & 0x80000000) | (m[(k + 1) % 624] & 0x7fffffff);
        let v = m[(k + 397) % 624] ^ (y >>> 1);
        if (y & 1) v ^= 0x9908b0df;
        m[k] = v >>> 0;
      }
      this.i = 0;
    }
    u32() {
      if (this.i >= 624) this.gen();
      let y = this.mt[this.i++];
      y ^= y >>> 11; y ^= (y << 7) & 0x9d2c5680; y ^= (y << 15) & 0xefc60000; y ^= y >>> 18;
      return y >>> 0;
    }
    dbl() {
      const a = this.u32() >>> 5, b = this.u32() >>> 6;
      return (a * 67108864 + b) / 9007199254740992;
    }
    randn() {
      if (this.hasG) { this.hasG = false; return this.g; }
      let x1, x2, r2;
      do { x1 = 2 * this.dbl() - 1; x2 = 2 * this.dbl() - 1; r2 = x1 * x1 + x2 * x2; } while (r2 >= 1 || r2 === 0);
      const f = Math.sqrt(-2 * Math.log(r2) / r2);
      this.g = f * x1; this.hasG = true;
      return f * x2;
    }
    interval(max) {
      if (max === 0) return 0;
      let m = max; m |= m >>> 1; m |= m >>> 2; m |= m >>> 4; m |= m >>> 8; m |= m >>> 16;
      let v;
      do { v = (this.u32() & m) >>> 0; } while (v > max);
      return v;
    }
    shuffle(a) {
      for (let i = a.length - 1; i > 0; i--) {
        const j = this.interval(i); [a[i], a[j]] = [a[j], a[i]];
      }
    }
  }
  function sample(rng, context) {
    const result = [];
    while (result.length < negatives) {
      const u = rng.dbl();
      let lo = 0, hi = cdf.length;
      while (lo < hi) {
        const mid = (lo + hi) >> 1;
        if (cdf[mid] <= u) lo = mid + 1; else hi = mid;
      }
      if (lo !== context) result.push(lo);
    }
    return result;
  }
  function predict(frame, center, negative) {
    const scores = frame.W.map(u => dot(frame.E[center], u));
    return { scores, values: negative ? scores.map(sigmoid) : softmax(scores) };
  }
  function softmaxStep(E, W, c, x, rate) {
    const h = E[c].slice(), u = W[x].slice();
    const scores = W.map(row => dot(h, row)), p = softmax(scores);
    const error = p.slice(); error[x] -= 1;
    const gh = h.map((_, i) => W.reduce((total, row, j) => total + row[i] * error[j], 0));
    const gu = h.map(v => v * error[x]);
    for (let i = 0; i < dimensions; i++) {
      for (let j = 0; j < words.length; j++) W[j][i] -= rate * h[i] * error[j];
      E[c][i] -= rate * gh[i];
    }
    const after = softmax(W.map(row => dot(E[c], row)))[x];
    return {
      word: x, label: 1, h, u, gh, gu, probability: p[x], after,
      loss: -Math.log(p[x] + 1e-12), afterLoss: -Math.log(after + 1e-12),
      hAfter: E[c].slice(), uAfter: W[x].slice()
    };
  }
  function binaryStep(E, W, c, x, label, rate) {
    const h = E[c].slice(), u = W[x].slice(), score = dot(h, u);
    const probability = sigmoid(score), error = probability - label;
    const gh = u.map(v => error * v), gu = h.map(v => error * v);
    for (let i = 0; i < dimensions; i++) { E[c][i] -= rate * gh[i]; W[x][i] -= rate * gu[i]; }
    const afterScore = dot(E[c], W[x]);
    return {
      word: x, label, h, u, gh, gu, probability, after: sigmoid(afterScore),
      loss: softplus(label ? -score : score), afterLoss: softplus(label ? -afterScore : afterScore),
      hAfter: E[c].slice(), uAfter: W[x].slice()
    };
  }
  function train({ negative = false, rate = 0.05, epochs = maxEpochs } = {}) {
    if (typeof negative !== 'boolean' || !Number.isFinite(rate) || rate < 0 || rate > 1 ||
        !Number.isInteger(epochs) || epochs < 1 || epochs > maxEpochs) {
      throw new RangeError('Choose a valid objective, learning rate from 0 to 1, and 1 to 200 epochs.');
    }
    const rng = new NpRandom(42);
    const E = words.map(() => Array.from({ length: dimensions }, () => rng.randn() * 0.1));
    const W = words.map(() => Array.from({ length: dimensions }, () => rng.randn() * 0.1));
    const order = pairs.map(p => p.slice()), frames = [{ epoch: 0, E: copy(E), W: copy(W), loss: null }];
    for (let epoch = 1; epoch <= epochs; epoch++) {
      rng.shuffle(order);
      let loss = 0;
      for (const [c, x] of order) {
        if (negative) {
          const noise = sample(rng, x);
          let pairLoss = binaryStep(E, W, c, x, 1, rate).loss;
          for (const n of noise) pairLoss += binaryStep(E, W, c, n, 0, rate).loss;
          loss += pairLoss;
        } else loss += softmaxStep(E, W, c, x, rate).loss;
      }
      if (![loss, ...E.flat(), ...W.flat()].every(Number.isFinite)) {
        throw new Error(`Training produced a non-finite value at epoch ${epoch}. Choose a smaller learning rate.`);
      }
      frames.push({ epoch, E: copy(E), W: copy(W), loss: loss / pairs.length });
    }
    return frames;
  }
  function preview(frame, center, context, negative, rate) {
    const E = copy(frame.E), W = copy(frame.W);
    const noise = negative ? sample(new NpRandom(42), context) : [];
    const updates = negative ? [binaryStep(E, W, center, context, 1, rate)] : [softmaxStep(E, W, center, context, rate)];
    for (const n of noise) updates.push(binaryStep(E, W, center, n, 0, rate));
    return { updates, E, W, noise };
  }
  function neighbors(center) {
    const result = new Map();
    for (const [c, x] of pairs) if (c === center) result.set(x, (result.get(x) || 0) + 1);
    return result;
  }
  function inspectWeight(frame, center, context, negative) {
    return [-0.01, 0, 0.01].map(change => {
      const W = copy(frame.W);
      W[context][0] += change;
      const prediction = predict({ E: frame.E, W }, center, negative);
      return {
        change, weight: W[context][0],
        loss: negative ? softplus(-prediction.scores[context]) : -Math.log(prediction.values[context] + 1e-12)
      };
    });
  }
  return { corpus, words, pairs, counts, dimensions, negatives, maxEpochs, train, predict, preview, neighbors, softplus, inspectWeight };
})();

if (typeof document !== 'undefined') {
  const lesson = Word2VecIntuition;
  const objective = document.body.dataset.objective;
  if (!['softmax', 'negative'].includes(objective)) throw new Error('This lesson needs a chapter-specific objective.');
  const negative = objective === 'negative';
  const $ = id => document.getElementById(id);
  const escape = text => String(text).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/"/g, '&quot;');
  const number = value => Math.abs(value) > 0 && Math.abs(value) < 0.0001 ? value.toExponential(3) : value.toFixed(4);
  const percent = value => `${(value * 100).toFixed(2)}%`;
  let frames, epoch = 0, rate = 0.05, timer = null;
  $('center').innerHTML = lesson.words.map(w => `<option>${escape(w)}</option>`).join('');
  $('center').value = 'cats';
  $('corpus').textContent = lesson.corpus.join('\n');
  function stop() {
    if (timer !== null) clearInterval(timer);
    timer = null; $('play').textContent = 'Play training'; $('play').setAttribute('aria-pressed', 'false');
  }
  function setContexts() {
    const c = lesson.words.indexOf($('center').value);
    $('context').innerHTML = [...lesson.neighbors(c)].map(([x]) => `<option>${escape(lesson.words[x])}</option>`).join('');
  }
  function drawLoss() {
    const width = 600, height = 300, left = 65, right = 22, top = 24, bottom = 48;
    const uniform = negative ? 4 * Math.log(2) : Math.log(lesson.words.length);
    const prior = 1 / (lesson.negatives + 1);
    const low = negative ? -Math.log(prior) - lesson.negatives * Math.log(1 - prior) : entropyFloor();
    const high = Math.max(uniform, ...frames.slice(1).map(f => f.loss)) * 1.08;
    const x = e => left + (width - left - right) * e / lesson.maxEpochs;
    const y = v => top + (height - top - bottom) * (high - v) / high;
    let svg = `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Average training loss through epoch ${epoch}">
      <title>Loss through epoch ${epoch}</title><desc>Each point averages the losses recorded during one epoch. Lower means a better fit to this training objective; the curve need not decrease at every epoch.</desc>`;
    for (let i = 0; i <= 4; i++) {
      const value = high * i / 4;
      svg += `<line x1="${left}" x2="${width-right}" y1="${y(value)}" y2="${y(value)}" stroke="var(--line)"/>
        <text x="${left-10}" y="${y(value)+5}" text-anchor="end">${value.toFixed(2)}</text>`;
    }
    for (const e of [0, 50, 100, 150, 200]) svg += `<text x="${x(e)}" y="${height-24}" text-anchor="middle">${e}</text>`;
    svg += `<text x="${left}" y="16">mean loss</text><text x="${width/2}" y="${height-3}" text-anchor="middle">epoch</text>`;
    svg += `<line x1="${left}" x2="${width-right}" y1="${y(uniform)}" y2="${y(uniform)}" stroke="var(--muted)" stroke-dasharray="5 5"/>`;
    svg += `<line x1="${left}" x2="${width-right}" y1="${y(low)}" y2="${y(low)}" stroke="var(--target)" stroke-dasharray="3 5"/>`;
    const points = frames.slice(1, epoch + 1);
    if (points.length) {
      const path = points.map((f, i) => `${i ? 'L' : 'M'}${x(f.epoch)},${y(f.loss)}`).join(' ');
      svg += `<path d="${path}" stroke="var(--center)" stroke-width="3" fill="none"/>
        <circle cx="${x(epoch)}" cy="${y(frames[epoch].loss)}" r="5" fill="var(--center)"/>`;
    } else svg += `<text x="${width/2}" y="${height/2}" text-anchor="middle">Advance an epoch to see measured loss.</text>`;
    $('lossPlot').innerHTML = svg + '</svg>';
    $('lossReference').textContent = negative
      ? `Grey dashed line: ${uniform.toFixed(4)} for outputs of 0.5. Ochre dotted line: ${low.toFixed(4)} for a constant output of 0.25, using only the 1:3 label ratio without distinguishing any words. Loss is summed over four examples, then averaged over observed pairs.`
      : `Grey dashed line: ${uniform.toFixed(4)} for uniform predictions. Ochre dotted line: ${low.toFixed(4)}, the best fixed-distribution average for these pairs. This is a reference, not a strict lower bound on the online losses collected while weights change.`;
  }
  function entropyFloor() {
    let total = 0;
    for (let c = 0; c < lesson.words.length; c++) {
      const counts = [...lesson.neighbors(c).values()], sum = counts.reduce((a, b) => a + b, 0);
      for (const n of counts) total -= n * Math.log(n / sum);
    }
    return total / lesson.pairs.length;
  }
  function predictionRow(j, values, observed) {
    return `<div class="prediction-row${observed.has(j) ? ' observed' : ''}">
      <span class="word">${escape(lesson.words[j])}${observed.has(j) ? ' *' : ''}</span>
      <span class="bar"><i style="width:${values[j]*100}%"></i></span><span class="number">${percent(values[j])}</span></div>`;
  }
  function render() {
    const frame = frames[epoch], c = lesson.words.indexOf($('center').value), x = lesson.words.indexOf($('context').value);
    const observed = lesson.neighbors(c), prediction = lesson.predict(frame, c, negative);
    const initial = lesson.predict(frames[0], c, negative);
    const ranked = lesson.words.map((_, j) => j).sort((a, b) => prediction.values[b] - prediction.values[a]);
    const shown = ranked.filter((j, i) => i < 5 || observed.has(j));
    $('predictions').innerHTML = shown.map(j => predictionRow(j, prediction.values, observed)).join('');
    $('otherPredictions').innerHTML = ranked.filter(j => !shown.includes(j)).map(j => predictionRow(j, prediction.values, observed)).join('');
    $('trainingStatus').textContent = `Epoch ${epoch} of ${lesson.maxEpochs}. ${epoch * lesson.pairs.length} observed pairs processed.`;
    $('epoch').value = epoch;
    $('epochValue').textContent = `${epoch} / ${lesson.maxEpochs}`;
    $('meanLoss').textContent = frame.loss === null ? 'Not trained yet' : number(frame.loss);
    $('firstLoss').textContent = epoch ? `Epoch 1: ${number(frames[1].loss)}` : 'One epoch means all 108 training pairs.';
    $('next').disabled = epoch === lesson.maxEpochs;
    $('play').disabled = epoch === lesson.maxEpochs;
    $('end').disabled = epoch === lesson.maxEpochs;
    $('predictionExplanation').textContent = negative
      ? `For "${lesson.words[c]}", each bar is a separate data-versus-noise sigmoid output. They do not add up to 100%. We inspect all 21 candidates here; training scores only the observed word and three sampled words per pair.`
      : `Given only "${lesson.words[c]}", the model scores all 21 words and turns those scores into probabilities that add up to 100%. Words marked * really occur beside this input in the corpus.`;
    $('observedContexts').textContent = [...observed].map(([j, n]) => `${lesson.words[j]} (${n})`).join(', ');
    $('pairName').textContent = `${lesson.words[c]} / ${lesson.words[x]}`;
    $('currentOutput').textContent = percent(prediction.values[x]);
    $('initialOutput').textContent = percent(initial.values[x]);
    $('currentScore').textContent = number(prediction.scores[x]);
    $('selectedVector').textContent = `[${frame.E[c].map(number).join(', ')}]`;
    const h = frame.E[c], u = frame.W[x];
    $('vectorLookup').innerHTML = `Input <b>${escape(lesson.words[c])}</b> has ID c = ${c} and looks up <code>E[c] = [${h.map(number).join(', ')}]</code>.
      Candidate <b>${escape(lesson.words[x])}</b> has ID x = ${x} and uses <code>${negative ? 'W[x]' : 'W[:, x]'} = [${u.map(number).join(', ')}]</code>.
      These are the current weights at epoch ${epoch}, not hand-chosen example vectors.`;
    $('dotRows').innerHTML = h.map((v, i) => `<tr><th scope="row">${i + 1}</th><td class="number">${number(v)}</td>
      <td class="number">${number(u[i])}</td><td class="number">${number(v * u[i])}</td></tr>`).join('');
    if (negative) {
      const clipped = Math.max(-15, Math.min(15, prediction.scores[x]));
      $('transformExample').innerHTML = `<p>For this pair, <code>q = 1 / (1 + exp(-(${number(clipped)}))) = ${number(prediction.values[x])}</code>.
        This is ${percent(prediction.values[x])} assigned to label 1, and ${percent(1 - prediction.values[x])} assigned to label 0.
        ${clipped !== prediction.scores[x] ? 'The helper clipped the score to [-15, 15] before applying sigmoid.' : 'No score clipping is active for this pair.'}</p>`;
    } else {
      const max = Math.max(...prediction.scores), exp = prediction.scores.map(s => Math.exp(s - max));
      const sum = exp.reduce((a, b) => a + b, 0);
      const candidates = [...new Set([x, ranked.find(j => j !== x), ranked.at(-1)])];
      $('transformExample').innerHTML = `<div class="scroll" tabindex="0" role="region" aria-label="Turn the actual scores into probabilities"><table>
        <caption>Three candidates shown; the denominator includes all 21 words</caption>
        <thead><tr><th scope="col">Candidate</th><th scope="col">Score</th><th scope="col">exp(score - maximum)</th><th scope="col">Divide by ${number(sum)}</th></tr></thead>
        <tbody>${candidates.map(j => `<tr><th scope="row">${escape(lesson.words[j])}</th><td class="number">${number(prediction.scores[j])}</td>
          <td class="number">${number(exp[j])}</td><td class="number">${percent(prediction.values[j])}</td></tr>`).join('')}</tbody></table></div>
        <p class="note">The largest score is ${number(max)}. Subtracting it before exponentiating keeps numbers safe without changing the probabilities. Values are rounded here; calculations use full precision.</p>`;
    }
    const probability = prediction.values[x];
    const observedLoss = negative ? lesson.softplus(-prediction.scores[x]) : -Math.log(probability + 1e-12);
    $('observedLoss').textContent = number(observedLoss);
    $('lossMeaning').textContent = negative
      ? `For the selected observed pair "${lesson.words[c]} / ${lesson.words[x]}", the model assigns ${percent(probability)} to label 1. Turning that assigned probability into a cost gives ${number(observedLoss)}. A sampled label-0 example would reward the opposite prediction.`
      : `The real example asks for "${lesson.words[x]}". The model gives it ${percent(probability)}, producing loss ${number(observedLoss)}. This is one example's loss, not the epoch average in the chart.`;
    if (negative) {
      $('noiseLoss').textContent = number(lesson.softplus(prediction.scores[x]));
      $('noiseMeaning').textContent = `If this same pair were sampled as noise in another training example, the label would be 0 instead. Its loss would then be ${number(lesson.softplus(prediction.scores[x]))}. The label comes from how the example was obtained, not a permanent truth about the two words.`;
    }
    const result = lesson.preview(frame, c, x, negative, rate), first = result.updates[0];
    $('updateRows').innerHTML = result.updates.map(u => `<tr><td>${escape(lesson.words[u.word])}${negative ? ` (label ${u.label})` : ''}</td>
      <td class="number">${percent(u.probability)}</td><td class="number">${percent(u.after)}</td>
      <td class="number">${number(u.loss)}</td><td class="number">${number(u.afterLoss)}</td></tr>`).join('');
    $('weightChange').textContent = `${number(first.u[0])} - ${rate} × (${number(first.gu[0])}) = ${number(first.uAfter[0])}`;
    $('gradientMeaning').textContent = `The gradient for this output weight is ${number(first.gu[0])}. ${
      first.gu[0] < 0 ? 'A negative gradient means increasing this weight is the local downhill direction.'
        : first.gu[0] > 0 ? 'A positive gradient means decreasing this weight is the local downhill direction.'
          : 'A zero gradient gives no change for this weight.'} The update also changes the input vector; neither table can be ignored.`;
    $('gradientChain').textContent = `For the selected observed example: score feedback = ${number(first.probability)} - 1 = ${number(first.probability - 1)}.
      This output weight's multiplier in the dot product is the input coordinate ${number(first.h[0])}.
      Their product is ${number(first.h[0])} × (${number(first.probability - 1)}) = ${number(first.gu[0])}, the gradient for this weight.
      The input coordinate's gradient is ${number(first.gh[0])}${negative ? ', using the output vector as its multiplier' : ', combining contributions from all 21 candidate scores'}.`;
    $('sensitivityRows').innerHTML = lesson.inspectWeight(frame, c, x, negative).map(v => `<tr>
      <th scope="row">${v.change < 0 ? 'Lower by 0.01' : v.change > 0 ? 'Raise by 0.01' : 'Leave unchanged'}</th>
      <td class="number">${v.weight.toFixed(7)}</td><td class="number">${v.loss.toFixed(7)}</td></tr>`).join('');
    $('rateValue').textContent = rate;
    $('trainingResult').textContent = `For "${lesson.words[c]} / ${lesson.words[x]}", the output changes from ${percent(initial.values[x])} before training to ${percent(probability)} at epoch ${epoch}. ${
      negative ? 'This is its data-versus-noise output, not a probability over context words.' : 'This is its context probability; other real neighbours must also receive probability.'}`;
    $('updateInterpretation').textContent = rate === 0
      ? 'The learning rate is zero: the model calculates feedback but changes no weights. Compare the predictions with epoch 0; they stay the same.'
      : `For the observed example, this isolated step ${first.afterLoss < first.loss ? 'lowers' : first.afterLoss > first.loss ? 'raises' : 'leaves unchanged'} its loss from ${number(first.loss)} to ${number(first.afterLoss)}. A gradient describes an infinitesimal change; a finite step, especially a large one, need not help every example.`;
    if (negative) $('previewNoise').textContent = `Preview noise: ${result.noise.map(j => lesson.words[j]).join(', ')}. This fixed sample uses count^0.75 and excludes only "${lesson.words[x]}"; duplicates and other real neighbours are allowed. It is not the run's next random draw. Each row uses the weights immediately after the preceding row.`;
    const norm = row => Math.sqrt(row.reduce((s, v) => s + v * v, 0));
    const nearest = lesson.words.map((w, j) => ({ w, j, similarity: frame.E[c].reduce((s, v, i) => s + v * frame.E[j][i], 0) / (norm(frame.E[c]) * norm(frame.E[j])) }))
      .filter(v => v.j !== c).sort((a, b) => b.similarity - a.similarity).slice(0, 3);
    $('similarWords').textContent = nearest.map(v => `${v.w} (${v.similarity.toFixed(3)})`).join(', ');
    $('rateMeaning').textContent = rate === 0
      ? `No learning: weights and predictions stay fixed.${negative ? ' The loss can still wiggle because fresh noise words are sampled.' : ''}`
      : 'Changing the learning rate starts a fresh run with the same initialization. A bigger step is not automatically a better step.';
    $('convergence').textContent = epoch < 2
      ? 'Watch the trend over many epochs, not a single update. The chart records loss during training, not a test-set score.'
      : `Mean loss is ${number(frame.loss)}, compared with ${number(frames[1].loss)} in epoch 1. ${
        Math.abs(frame.loss - frames[epoch - 1].loss) < 0.001
          ? 'The last two epoch means differ by less than 0.001. That is a local plateau, not proof that the best solution has been found.'
          : 'The curve is still changing. A downward trend means the model is fitting this training task better, not that every prediction is improving.'}`;
    drawLoss();
  }
  function startRun() {
    stop(); epoch = 0; rate = Number($('rate').value); $('error').hidden = true;
    frames = lesson.train({ negative, rate }); render();
  }
  $('center').onchange = () => { setContexts(); render(); };
  $('context').onchange = render;
  $('rate').onchange = startRun;
  $('epoch').oninput = () => { stop(); epoch = Number($('epoch').value); render(); };
  $('next').onclick = () => { stop(); epoch = Math.min(epoch + 1, lesson.maxEpochs); render(); };
  $('end').onclick = () => { stop(); epoch = lesson.maxEpochs; render(); };
  $('reset').onclick = () => { stop(); epoch = 0; render(); };
  $('play').onclick = () => {
    if (timer !== null) { stop(); return; }
    $('play').textContent = 'Pause training'; $('play').setAttribute('aria-pressed', 'true');
    timer = setInterval(() => {
      epoch = Math.min(epoch + 1, lesson.maxEpochs); render();
      if (epoch === lesson.maxEpochs) stop();
    }, 150);
  };
  window.addEventListener('pagehide', stop);
  window.addEventListener('error', e => {
    stop(); $('error').hidden = false; $('error').textContent = `The lesson could not finish: ${e.message}. Reload to restart.`;
  });
  setContexts(); startRun();
}
