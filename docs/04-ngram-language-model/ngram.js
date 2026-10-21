// Page logic for chapter 04. The models themselves live in ngram-engine.js.
(() => {
  const M = NgramEngine;
  const $ = id => document.getElementById(id);
  const esc = value => String(value).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  const word = id => M.vocab[id];
  const f = (value, digits = 4) => (Object.is(Math.round(value * 10 ** digits), -0) ? 0 : value).toFixed(digits);
  const pct = value => `${(value * 100).toFixed(1)}%`;
  const showError = message => { $('loadError').hidden = false; $('loadError').textContent = message; };
  const store = {
    get(key) { try { return localStorage.getItem(key); } catch { return null; } },
    set(key, value) { try { localStorage.setItem(key, value); } catch { /* the page works without saving */ } },
  };

  function bars(rows, total, bestId) {
    const max = Math.max(...rows.map(r => r.count));
    return `<div class="count-bars">${rows.map(r => `<div class="count-bar${r.id === bestId ? ' best' : ''}">
      <span class="word">${esc(word(r.id))}</span><span class="bar"><i style="width:${(100 * r.count / max).toFixed(1)}%"></i></span>
      <span class="number">${r.count} / ${total} = ${f(r.count / total, 3)}</span></div>`).join('')}</div>`;
  }

  // ---------------- intuition page ----------------
  function intuition() {
    let position = 0;
    const shown = M.tokens.slice(0, 26);
    function renderWindow() {
      const n = Number($('windowN').value);
      if (position + n > shown.length) position = 0;
      $('windowStream').innerHTML = shown.map((w, i) => {
        const cls = i >= position && i < position + n - 1 ? 'ctx' : i === position + n - 1 ? 'tgt' : w === '<eos>' ? 'eos' : '';
        return `<span class="${cls}">${esc(w)}</span>`;
      }).join('');
      const context = shown.slice(position, position + n - 1).join(' ');
      $('windowExample').innerHTML = `Context <span class="c-ctx">${esc(context)}</span> → target <span class="c-tgt">${esc(shown[position + n - 1])}</span>`;
    }
    $('windowN').addEventListener('change', () => { position = 0; renderWindow(); });
    $('windowNext').addEventListener('click', () => { position += 1; renderWindow(); });
    renderWindow();

    const options = M.vocab.filter(w => w !== '<eos>').map(w => `<option>${esc(w)}</option>`).join('');
    $('countA').innerHTML = options; $('countB').innerHTML = options;
    $('countA').value = store.get('ngramCountA') || 'the';
    $('countB').value = store.get('ngramCountB') || 'cat';
    function renderCount() {
      const a = $('countA').value, b = $('countB').value;
      store.set('ngramCountA', a); store.set('ngramCountB', b);
      const result = M.countLookup([M.wordToId.get(a), M.wordToId.get(b)]);
      let html = '';
      if (result.used < 2) {
        html += '<div class="backoff">' + result.steps.map(step => {
          const label = step.context.length ? `<code>${esc(step.context.map(word).join(' '))}</code>` : 'no context (plain word counts)';
          return step.total ? `<p>${label}: seen ${step.total} time${step.total > 1 ? 's' : ''}. Use these counts.</p>`
            : `<p>${label}: never seen in the 10 sentences. Drop the oldest word.</p>`;
        }).join('') + '</div>';
      }
      const rows = result.next.map((count, id) => ({ id, count })).filter(r => r.count).sort((x, y) => y.count - x.count);
      const used = result.steps[result.steps.length - 1].context.map(word).join(' ');
      html += `<p>Words that came after <code>${esc(used || '(anything)')}</code>, with count ÷ total:</p>` + bars(rows, result.total, result.prediction);
      html += `<p>Guess: <b class="c-tgt">${esc(word(result.prediction))}</b>${rows.length > 1 && rows[1].count === rows[0].count ? ' (a tie: the script keeps the word it saw first)' : ''}.</p>`;
      $('countResult').innerHTML = html;
    }
    $('countA').addEventListener('change', renderCount); $('countB').addEventListener('change', renderCount);
    renderCount();

    fetch('results.json').then(r => {
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      return r.json();
    }).then(renderResults).catch(error => showError(`Could not load the Python results (results.json): ${error.message}. Open the page through a web server, for example python -m http.server --directory docs.`));
  }

  function renderResults(data) {
    const rows = data.results;
    $('countRows').innerHTML = rows.map(r => `<tr><td>${r.n}</td><td>${r.n - 1}</td><td class="number">${pct(r.fullContextSeen)}</td><td class="number">${pct(r.countTrainAccuracy)}</td><td class="number">${pct(r.countTestAccuracy)}</td></tr>`).join('');
    const first = rows[0], last = rows[rows.length - 1];
    const bestCount = rows.reduce((a, b) => (b.countTestAccuracy > a.countTestAccuracy ? b : a));
    $('countReading').innerHTML = `From n = ${first.n} to n = ${last.n}, accuracy on the training text rises from ${pct(first.countTrainAccuracy)} to ${pct(last.countTrainAccuracy)}: the model is memorising. On new text, the full context had been seen only ${pct(last.fullContextSeen)} of the time at n = ${last.n}, and accuracy peaks at n = ${bestCount.n} (${pct(bestCount.countTestAccuracy)}).`;
    $('compareRows').innerHTML = rows.map(r => `<tr><td>${r.n}</td><td class="number">${pct(r.countTestAccuracy)}</td><td class="number">${pct(r.neuralTestAccuracy)}</td><td class="number">${r.neuralSeconds.toFixed(0)} s</td></tr>`).join('');
    const s = data.settings;
    $('compareReading').textContent = `Neural model: vocabulary ${s.vocabSize.toLocaleString()}, D = ${s.D}, ${s.epochs} epoch${s.epochs > 1 ? 's' : ''} over ${s.trainWords.toLocaleString()} training words, batches of ${s.batchSize}, learning rate ${s.learningRate}. Times are from the run that made this page; yours will differ.`;

    const prompts = [...new Set(data.samples.map(x => x.prompt))];
    $('genN').innerHTML = rows.map(r => `<option value="${r.n}">${r.n}</option>`).join('');
    $('genPrompt').innerHTML = prompts.map(p => `<option>${esc(p)}</option>`).join('');
    const saved = (store.get('ngramGen') || '').split('|');
    if (saved.length === 4) {
      [$('genN').value, $('genModel').value, $('genMethod').value, $('genPrompt').value] = saved;
      if (!$('genN').value) $('genN').value = String(rows[0].n);
      if (!$('genPrompt').value) $('genPrompt').value = prompts[0];
    } else {
      $('genN').value = String(rows[0].n);
    }
    const renderSample = () => {
      const pick = [$('genN').value, $('genModel').value, $('genMethod').value, $('genPrompt').value];
      store.set('ngramGen', pick.join('|'));
      const sample = data.samples.find(x => String(x.n) === pick[0] && x.model === pick[1] && x.method === pick[2] && x.prompt === pick[3]);
      if (!sample) { $('genText').textContent = 'No sample for this choice.'; return; }
      const words = sample.text.replace(/\n\n/g, ' ↵ ').split(/\s+/).filter(Boolean);
      const promptLength = sample.prompt.split(' ').length;
      // mark every word that belongs to a 4-word phrase appearing more than once
      const seen = new Map(), marked = new Array(words.length).fill(false);
      for (let i = 0; i + 4 <= words.length; i++) {
        const key = words.slice(i, i + 4).join(' ');
        if (!seen.has(key)) seen.set(key, []);
        seen.get(key).push(i);
      }
      for (const starts of seen.values()) if (starts.length > 1) starts.forEach(i => { for (let k = i; k < i + 4; k++) marked[k] = true; });
      const share = marked.filter(Boolean).length / words.length;
      $('genText').innerHTML = words.map((w, i) => {
        const text = esc(w);
        if (i < promptLength) return `<span class="prompt">${text}</span>`;
        return marked[i] ? `<mark>${text}</mark>` : text;
      }).join(' ');
      $('genRepeat').textContent = share > 0
        ? `Highlighted: words inside a 4-word phrase that appears more than once (${pct(share)} of the text). The prompt is in teal. ↵ marks the end of a paragraph.`
        : 'No 4-word phrase repeats in this text. The prompt is in teal. ↵ marks the end of a paragraph.';
    };
    ['genN', 'genModel', 'genMethod', 'genPrompt'].forEach(id => $(id).addEventListener('change', renderSample));
    renderSample();
  }

  // ---------------- math & code page ----------------
  const PHASES = [
    { key: 'lookup', title: '1. Look up the context embeddings', lines: ['x = model["E"][contexts]'],
      why: 'A word ID is only a name. To compute with a word we need numbers, so we copy each context word’s row out of the embedding table E.' },
    { key: 'join', title: '2. Join them into one list x', lines: ['x = model["E"][contexts]', 'batch_size, context_length = contexts.shape'],
      why: 'The output layer needs one list of numbers per window. Putting the rows end to end keeps the order: the first D numbers always come from the older word, the next D from the newer word. So the model can tell “dog bit” from “bit dog”.' },
    { key: 'scores', title: '3. Score every word in the vocabulary', lines: ['scores = x @ model["W"] + model["b"]'],
      why: 'Each word v has its own column of W. Its score multiplies the 4 numbers of x by the 4 numbers of that column, adds them up, and adds the word’s starting score b. A high score means “this word fits after this context”.' },
    { key: 'softmax', title: '4. Turn the scores into probabilities', lines: ['probabilities = softmax(scores)', 'shifted = scores - scores.max(', 'exp_scores = np.exp(shifted)', 'return exp_scores / exp_scores.sum('],
      why: 'Scores can be any number, negative too. exp makes every score positive, and dividing by the total makes them add up to 1, so we can read them as “how likely is each next word”. Subtracting the largest score first does not change the result, because it divides out, but it stops exp from overflowing.' },
    { key: 'loss', title: '5. Measure the mistake: cross-entropy loss', lines: ['loss = -np.log(probabilities[rows, targets]'],
      why: 'We only look at the probability the model gave to the real next word. If it was 1, the loss −ln(1) is 0. The smaller it was, the bigger the loss. The 1e-12 only prevents log(0).' },
    { key: 'dscores', title: '6. Gradient of the loss for each score', lines: ['d_scores = probabilities', 'd_scores[rows, targets] -= 1.0', 'd_scores /= batch_size'],
      why: 'This says which way to push each score. The real next word gets p − 1, which is negative: raising its score lowers the loss. Every other word gets p, which is positive: lowering its score lowers the loss. Here the batch has 1 window, so dividing by batch_size changes nothing.' },
    { key: 'dW', title: '7. Gradients for W and b', lines: ['d_W = x.T @ d_scores', 'd_b = d_scores.sum(axis=0)'],
      why: 'W[i, v] was multiplied by x[i] to make score v. So nudging W[i, v] changes score v by x[i] per unit, and its gradient is x[i] × d_scores[v]. b[v] was added straight to score v, so its gradient is just d_scores[v].' },
    { key: 'dx', title: '8. Gradient for x, then for E', lines: ['d_x = d_scores @ model["W"].T'],
      why: 'x[i] was used in every score, multiplied by W[i, v]. So its gradient adds W[i, v] × d_scores[v] over all words v. We use W from before the update. Then we cut d_x back into one piece of D numbers per context word: each piece belongs to that word’s row of E.' },
    { key: 'update', title: '9. Update W, b and the two rows of E', lines: ['model["W"] -= learning_rate * d_W', 'model["b"] -= learning_rate * d_b', 'np.add.at(model["E"], contexts'],
      why: 'Every number moves a small step against its gradient: new = old − learning rate × gradient. Only the rows of E for the context words change; every other word’s embedding stays the same. np.add.at makes sure a word that appears twice in the context gets both pieces added.' },
  ];

  function matrix(rows, { rowLabels, colLabels, caption, hotRows = [], hotCols = [], digits = 4 }) {
    let html = `<div class="scroll" tabindex="0" role="region" aria-label="${esc(caption)}"><table class="mat"><caption>${caption}</caption>`;
    if (colLabels) html += `<thead><tr><th></th>${colLabels.map(c => `<th scope="col">${esc(c)}</th>`).join('')}</tr></thead>`;
    html += '<tbody>' + rows.map((row, r) => `<tr>${rowLabels ? `<th scope="row">${esc(rowLabels[r])}</th>` : ''}${row.map((v, c) => {
      const cls = hotCols.includes(c) ? 'hot' : hotRows.includes(r) ? 'cool' : '';
      return `<td class="${cls}">${typeof v === 'number' ? f(v, digits) : esc(v)}</td>`;
    }).join('')}</tr>`).join('') + '</tbody></table></div>';
    return html;
  }

  function lineChart(values, { yLabel }) {
    const width = 640, height = 280, left = 64, right = 18, top = 20, bottom = 46;
    if (!values.length) return '<p class="note">Train to draw the loss curve.</p>';
    const maxY = Math.max(...values) * 1.05, minY = 0, maxX = Math.max(values.length, 2);
    const px = i => left + (width - left - right) * (i) / (maxX - 1 || 1);
    const py = v => top + (height - top - bottom) * (maxY - v) / (maxY - minY);
    let svg = `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Loss per epoch"><title>Average loss per epoch</title>`;
    for (let k = 0; k <= 4; k++) {
      const v = minY + (maxY - minY) * k / 4;
      svg += `<line x1="${left}" x2="${width - right}" y1="${py(v)}" y2="${py(v)}" stroke="var(--line)"/><text x="${left - 8}" y="${py(v) + 5}" text-anchor="end">${v.toFixed(1)}</text>`;
    }
    svg += `<text x="${(left + width - right) / 2}" y="${height - 6}" text-anchor="middle">epoch (1 to ${values.length})</text><text x="${left}" y="14">${esc(yLabel)}</text>`;
    svg += `<path d="${values.map((v, i) => `${i ? 'L' : 'M'}${px(i).toFixed(1)},${py(v).toFixed(1)}`).join(' ')}" fill="none" stroke="var(--center)" stroke-width="2.5"/>`;
    svg += `<circle cx="${px(values.length - 1)}" cy="${py(values[values.length - 1])}" r="4.5" fill="var(--target)"/></svg>`;
    return svg;
  }

  function math() {
    const N = 3, C = N - 1, D = M.D, V = M.V;
    const windows = M.windows(N);
    let model = M.initModel(N), applied = 0, selected = 0, phase = 0;

    $('vocabSize').textContent = V;
    $('vocabTable').innerHTML = `<div class="vocab-grid">${M.vocab.map((w, i) => `<span><b>${i}</b>${esc(w)} <span class="muted">×${M.counts.get(w)}</span></span>`).join('')}</div>`;
    $('windowCount').textContent = windows.length;
    $('batchSize').textContent = M.BATCH_SIZE;
    $('learningRate').textContent = M.LEARNING_RATE;
    $('windowList').innerHTML = windows.map((w, i) => `<button type="button" role="option" data-i="${i}" aria-selected="false">${esc(w.context.map(word).join(' '))} → ${esc(word(w.target))}</button>`).join('');
    $('windowList').addEventListener('click', event => {
      const button = event.target.closest('button[data-i]');
      if (button) { selected = Number(button.dataset.i); phase = 0; render(); }
    });

    // show only the functions the update runs, with their real line numbers
    const source = $('pySource').textContent.replace(/^\n/, '').split('\n');
    const highlighted = PythonSyntax.highlightLines(source.join('\n'));
    const wanted = ['def forward(', 'def softmax(', 'def train_step('];
    const blocks = wanted.map(start => {
      const from = source.findIndex(line => line.startsWith(start));
      let to = from + 1;
      while (to < source.length && !source[to].startsWith('def ') && !source[to].startsWith('#')) to++;
      while (to > from && source[to - 1].trim() === '') to--;
      return { from, to };
    }).filter(b => b.from >= 0);
    $('codeSource').innerHTML = blocks.map(({ from, to }) => source.slice(from, to).map((line, k) =>
      `<button type="button" class="source-line" data-line="${from + k}"><span class="line-number">${from + k + 1}</span><span>${highlighted[from + k] || ' '}</span></button>`).join('')).join('<div class="source-gap">⋮</div>');
    PythonSyntax.attachControls($('codeSource'));
    $('codeSource').addEventListener('click', event => {
      const button = event.target.closest('.source-line');
      if (!button) return;
      const text = source[Number(button.dataset.line)].trim();
      const found = PHASES.findIndex(p => p.lines.some(prefix => text.startsWith(prefix)));
      if (found >= 0) { phase = found; render(); }
    });

    $('phases').innerHTML = PHASES.map((p, i) => `<button type="button" data-p="${i}" aria-pressed="false">${esc(p.title.replace(/^\d+\. /, ''))}</button>`).join('');
    $('phases').addEventListener('click', event => {
      const button = event.target.closest('button[data-p]');
      if (button) { phase = Number(button.dataset.p); render(); }
    });
    $('prevPhase').addEventListener('click', () => { phase = Math.max(0, phase - 1); render(); });
    $('nextPhase').addEventListener('click', () => { phase = Math.min(PHASES.length - 1, phase + 1); render(); });
    $('applyUpdate').addEventListener('click', () => {
      M.trainStep(model, [windows[selected]]);
      applied += 1; selected = (selected + 1) % windows.length; phase = 0; render();
      $('windowList').querySelector(`[data-i="${selected}"]`)?.scrollIntoView({ block: 'nearest' });
    });
    $('resetModel').addEventListener('click', () => { model = M.initModel(N); applied = 0; phase = 0; render(); });

    function render() {
      const w = windows[selected], p = PHASES[phase];
      $('windowList').querySelectorAll('button').forEach(b => b.setAttribute('aria-selected', String(Number(b.dataset.i) === selected)));
      $('phases').querySelectorAll('button').forEach(b => b.setAttribute('aria-pressed', String(Number(b.dataset.p) === phase)));
      $('prevPhase').disabled = phase === 0; $('nextPhase').disabled = phase === PHASES.length - 1;
      $('modelStatus').textContent = applied
        ? `The weights already include ${applied} applied update${applied > 1 ? 's' : ''}. Window: ${w.context.map(word).join(' ')} → ${word(w.target)}.`
        : `Weights are the random start. Window: ${w.context.map(word).join(' ')} → ${word(w.target)}.`;
      $('codeSource').querySelectorAll('.source-line').forEach(line => {
        const text = source[Number(line.dataset.line)].trim();
        line.classList.toggle('active', p.lines.some(prefix => text.startsWith(prefix)));
      });
      $('codeSource').querySelector('.source-line.active')?.scrollIntoView({ block: 'nearest' });
      $('phaseTitle').textContent = p.title;
      $('phaseWhy').textContent = p.why;
      $('phaseWork').innerHTML = work(p.key, w);
      if (window.MathJax?.typesetPromise) MathJax.typesetPromise([$('phaseWork')]).catch(() => {});
    }

    function work(key, w) {
      const after = M.clone(model);
      const { dW, db, details } = M.trainStep(after, [w]);
      const d = details[0], t = w.target, ctxWords = w.context.map(word);
      const xLabels = w.context.flatMap((id, c) => Array.from({ length: D }, (_, k) => `${word(id)}[${k}]`));
      const colLabels = M.vocab;
      switch (key) {
        case 'lookup':
          return `<p>Context IDs: ${w.context.map((id, c) => `<code>${id}</code> (${esc(ctxWords[c])})`).join(' and ')}. Target ID <code>${t}</code> (${esc(word(t))}). The rows we copy are highlighted.</p>`
            + matrix(model.E, { rowLabels: M.vocab.map((v, i) => `${i} ${v}`), colLabels: ['E[·,0]', 'E[·,1]'], caption: `E, shape ${V} × ${D}`, hotRows: w.context })
            + `<div class="calc">${w.context.map((id, c) => `E[${id}] (${ctxWords[c]}) = [${model.E[id].map(v => f(v)).join(', ')}]`).join('\n')}</div>`;
        case 'join':
          return `<p>Shape goes from (1 window, ${C} words, ${D} numbers) to (1 window, ${C} × ${D} = ${C * D} numbers).</p>`
            + matrix([d.x], { colLabels: xLabels, caption: `x, ${C * D} numbers` })
            + `<div class="calc">${d.pieces.map((piece, c) => `piece ${c + 1} = E[${w.context[c]}] (${ctxWords[c]}) = [${piece.map(v => f(v)).join(', ')}]`).join('\n')}\nx = [${d.x.map(v => f(v)).join(', ')}]</div>`;
        case 'scores': {
          const terms = d.x.map((xi, i) => `${f(xi)} × ${f(model.W[i][t])}`).join('\n  + ');
          return matrix(model.W, { rowLabels: xLabels.map((l, i) => `${i}: ${l}`), colLabels, caption: `W, shape ${C * D} × ${V} (one column per word; the target's column is highlighted)`, hotCols: [t] })
            + matrix([model.b], { colLabels, caption: `b, ${V} starting scores`, hotCols: [t] })
            + `<p>For the target word <b>${esc(word(t))}</b> (column ${t}):</p><div class="calc">score[${esc(word(t))}] =   ${terms}\n  + b[${t}] = ${f(model.b[t])}\n  = ${f(d.scores[t])}</div>`
            + matrix([d.scores], { colLabels, caption: `scores = x @ W + b, all ${V} words`, hotCols: [t] });
        }
        case 'softmax': {
          const max = Math.max(...d.scores), exps = d.scores.map(s => Math.exp(s - max)), sum = exps.reduce((a, b) => a + b, 0);
          const rows = M.vocab.map((v, i) => [v, d.scores[i], d.scores[i] - max, exps[i], d.probabilities[i]]);
          return `<p>Largest score: ${f(max)}. Sum of all ${V} exp values: ${f(sum)}.</p>`
            + matrix(rows, { colLabels: ['word', 'score', 'score − max', 'exp(…)', 'probability = exp ÷ sum'], caption: 'Every word, every step', hotRows: [t] })
            + `<p>The probabilities add up to ${f(d.probabilities.reduce((a, b) => a + b, 0), 6)}. The real next word, <b>${esc(word(t))}</b>, got ${f(d.probabilities[t])}. A model that knew nothing would give every word 1/${V} = ${f(1 / V)}.</p>`;
        }
        case 'loss': {
          const loss = -Math.log(d.probabilities[t] + 1e-12);
          return `<div class="calc">loss = −ln(probability of ${word(t)})\n     = −ln(${f(d.probabilities[t], 6)})\n     = ${f(loss)}</div>`
            + `<p>For comparison: probability 1/${V} would cost −ln(1/${V}) = ${f(Math.log(V))}, and probability 0.9 would cost ${f(-Math.log(0.9))}.</p>`;
        }
        case 'dscores':
          return matrix(M.vocab.map((v, i) => [v, d.probabilities[i], i === t ? 1 : 0, d.dScores[i]]), { colLabels: ['word', 'probability', 'is target?', 'd_scores = p − (1 or 0)'], caption: 'Gradient for each score', hotRows: [t] })
            + `<p>Only <b>${esc(word(t))}</b> has a negative gradient (${f(d.dScores[t])}), so only its score will be pushed up. All the others are pushed down, each by as much as the probability it wrongly took.</p>`;
        case 'dW':
          return `<p>d_W is x (as a column) times d_scores (as a row): every pair (i, v) gets x[i] × d_scores[v].</p>`
            + matrix(dW, { rowLabels: xLabels.map((l, i) => `${i}: ${l}`), colLabels, caption: `d_W, shape ${C * D} × ${V}`, hotCols: [t] })
            + `<div class="calc">d_W[0, ${esc(word(t))}] = x[0] × d_scores[${esc(word(t))}] = ${f(d.x[0])} × ${f(d.dScores[t])} = ${f(dW[0][t])}</div>`
            + matrix([db], { colLabels, caption: 'd_b = d_scores (summed over a batch of 1)', hotCols: [t] });
        case 'dx': {
          const terms = M.vocab.map((v, i) => `${f(model.W[0][i], 3)} × ${f(d.dScores[i], 3)}`);
          const lines = [];
          for (let i = 0; i < terms.length; i += 3) lines.push(terms.slice(i, i + 3).join(' + '));
          return `<p>The first number of x, written out over all ${V} words (W[0, v] × d_scores[v]):</p><div class="calc">d_x[0] =   ${lines.join('\n         + ')}\n       = ${f(d.dX[0])}</div>`
            + matrix([d.dX], { colLabels: xLabels, caption: 'd_x, all 4 numbers' })
            + `<p>Cut into one piece per context word:</p><div class="calc">${w.context.map((id, c) => `gradient for E[${id}] (${ctxWords[c]}) = [${d.dX.slice(c * D, c * D + D).map(v => f(v)).join(', ')}]`).join('\n')}</div>`
            + (w.context[0] === w.context[1] ? '<p>Both context words are the same word, so both pieces are added to the same row.</p>' : '');
        }
        case 'update': {
          const lr = M.LEARNING_RATE, before = d.probabilities[t], next = M.forward(after, w.context).probabilities[t];
          return `<p>Learning rate η = ${lr}. Every number of W and b changes; only rows ${w.context.join(' and ')} of E change.</p>`
            + `<div class="calc">${w.context.map((id, c) => `E[${id}] (${ctxWords[c]}): [${model.E[id].map(v => f(v)).join(', ')}] − ${lr} × [${d.dX.slice(c * D, c * D + D).map(v => f(v)).join(', ')}]${w.context.indexOf(id) !== c ? ' (added again)' : ''}`).join('\n')}\n${[...new Set(w.context)].map(id => `new E[${id}] = [${after.E[id].map(v => f(v)).join(', ')}]`).join('\n')}\n\nb[${t}] (${word(t)}): ${f(model.b[t])} − ${lr} × ${f(db[t])} = ${f(after.b[t])}</div>`
            + matrix(after.W, { rowLabels: xLabels.map((l, i) => `${i}: ${l}`), colLabels, caption: 'W after the update', hotCols: [t] })
            + `<p>Run the forward pass again with the new weights: the probability of <b>${esc(word(t))}</b> goes from ${f(before)} to <b>${f(next)}</b>, so the loss goes from ${f(-Math.log(before))} to ${f(-Math.log(next))}. Press “Apply this update” to keep it.</p>`;
        }
      }
      return '';
    }

    render();

    // ----- training section -----
    let trainModel = M.initModel(N), losses = [];
    function renderTraining() {
      $('lossPlot').innerHTML = lineChart(losses, { yLabel: 'average loss' });
      const { accuracy, loss } = M.evaluate(trainModel);
      $('lossReading').textContent = losses.length
        ? `Epoch ${losses.length}: training loss ${f(losses[losses.length - 1], 3)}. Started near −ln(1/${V}) = ${f(Math.log(V), 2)}, the loss of a model that gives every word the same probability.`
        : `Untrained. A random start gives roughly every word the same probability, so the loss is near −ln(1/${V}) = ${f(Math.log(V), 2)}.`;
      const probes = [['the', 'cat'], ['the', 'dog'], ['on', 'the'], ['bird', 'ate']];
      const top = ctx => {
        const probs = M.forward(trainModel, ctx.map(x => M.wordToId.get(x))).probabilities;
        return probs.map((p, i) => [i, p]).sort((a, b) => b[1] - a[1]).slice(0, 3).map(([i, p]) => `${esc(word(i))} ${f(p, 2)}`).join(', ');
      };
      const greedy = M.generateGreedy(ctx => {
        const probs = M.forward(trainModel, ctx).probabilities;
        return probs.indexOf(Math.max(...probs));
      }, ['the', 'dog'].map(x => M.wordToId.get(x)), C, 18).map(word).join(' ');
      $('trainedSummary').innerHTML = `<div class="metrics"><p class="note">Accuracy on all windows<strong>${pct(accuracy)}</strong></p><p class="note">Average loss now<strong>${f(loss, 3)}</strong></p></div>`
        + `<table><thead><tr><th scope="col">Context</th><th scope="col">Top 3 next words</th></tr></thead><tbody>${probes.map(c => `<tr><td><code>${c.join(' ')}</code></td><td>${top(c)}</td></tr>`).join('')}</tbody></table>`
        + `<p>Greedy text from <code>the dog</code>:</p><p class="generated" style="font-size:15px">${esc(greedy)}</p>`;
    }
    $('train1').addEventListener('click', () => { losses.push(M.trainEpoch(trainModel)); renderTraining(); });
    $('train50').addEventListener('click', () => { for (let i = 0; i < 50; i++) losses.push(M.trainEpoch(trainModel)); renderTraining(); });
    $('trainReset').addEventListener('click', () => { trainModel = M.initModel(N); losses = []; renderTraining(); });
    renderTraining();
  }

  document.addEventListener('DOMContentLoaded', () => {
    try {
      if (document.body.dataset.view === 'math') math(); else intuition();
    } catch (error) {
      showError(`Something went wrong while building this page: ${error.message}`);
      console.error(error);
    }
  });
})();
