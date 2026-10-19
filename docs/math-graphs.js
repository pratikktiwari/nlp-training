const MathGraphs = (() => {
  let sequence = 0;
  const escape = text => String(text).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/"/g, '&quot;');
  const format = value => value !== 0 && Math.abs(value) < 0.001
    ? value.toExponential(2) : String(Number(value.toFixed(4)));

  function slope({ title, context, x, xMin, xMax, xLabel, yLabel, curves }) {
    if (![x, xMin, xMax].every(Number.isFinite) || xMax <= xMin || x < xMin || x > xMax) {
      throw new RangeError('Slope graph requires a finite point inside an increasing domain.');
    }
    const series = curves.map((curve, index) => {
      const points = Array.from({ length: 121 }, (_, i) => {
        const at = xMin + (xMax - xMin) * i / 120;
        return [at, curve.fn(at)];
      });
      const value = curve.fn(x);
      if (![value, curve.slope, ...points.map(point => point[1])].every(Number.isFinite)) {
        throw new RangeError('Slope graph values and derivatives must be finite.');
      }
      return { ...curve, points, value, color: index === 0 ? 'var(--target)' : 'var(--center)' };
    });
    const values = series.flatMap(curve => [curve.value, ...curve.points.map(point => point[1])]);
    const min = Math.min(...values), max = Math.max(...values);
    const padding = Math.max((max - min) * 0.12, Math.abs(max) * 0.02, 0.0001);
    const yMin = yLabel === 'probability' ? 0 : Math.max(0, min - padding);
    const yMax = yLabel === 'probability' ? 1 : max + padding;
    const width = 640, height = 300, left = 78, right = 20, top = 34, bottom = 48;
    const px = value => left + (width - left - right) * (value - xMin) / (xMax - xMin);
    const py = value => top + (height - top - bottom) * (yMax - value) / (yMax - yMin);
    const clip = `slope-clip-${++sequence}`;
    let svg = `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${escape(title)}">
      <title>${escape(title)}</title><desc>Solid curves show ${escape(yLabel)} as ${escape(xLabel)} varies. Dots mark the selected value; dashed tangents show its local slope. ${escape(context)}</desc>
      <defs><clipPath id="${clip}"><rect x="${left}" y="${top}" width="${width-left-right}" height="${height-top-bottom}"/></clipPath></defs>`;
    for (let i = 0; i <= 4; i++) {
      const value = yMin + (yMax - yMin) * i / 4;
      svg += `<line x1="${left}" x2="${width-right}" y1="${py(value)}" y2="${py(value)}" stroke="var(--line)"/>
        <text x="${left-8}" y="${py(value)+4}" text-anchor="end">${format(value)}</text>`;
    }
    for (let i = 0; i <= 4; i++) {
      const value = xMin + (xMax - xMin) * i / 4;
      svg += `<text x="${px(value)}" y="${height-bottom+22}" text-anchor="middle">${format(value)}</text>`;
    }
    svg += `<text x="${left}" y="18" class="lbl">${escape(yLabel)}</text>
      <text x="${(left+width-right)/2}" y="${height-4}" text-anchor="middle" class="lbl">${escape(xLabel)}</text>
      <line x1="${px(x)}" x2="${px(x)}" y1="${top}" y2="${height-bottom}" stroke="var(--line-strong)" stroke-dasharray="3 4"/>`;
    svg += `<g clip-path="url(#${clip})">`;
    for (const curve of series) {
      const path = curve.points.map(([at, value], i) => `${i ? 'L' : 'M'}${px(at)},${py(value)}`).join(' ');
      const span = Math.min((xMax-xMin)*0.14, (yMax-yMin)*0.18/Math.max(Math.abs(curve.slope), 1e-12));
      const a = Math.max(xMin, x-span), b = Math.min(xMax, x+span);
      svg += `<path d="${path}" fill="none" stroke="${curve.color}" stroke-width="2.5"/>
        <line x1="${px(a)}" y1="${py(curve.value+curve.slope*(a-x))}" x2="${px(b)}" y2="${py(curve.value+curve.slope*(b-x))}" stroke="var(--fg)" stroke-width="2" stroke-dasharray="7 5"/>
        <circle cx="${px(x)}" cy="${py(curve.value)}" r="5" fill="${curve.color}" stroke="var(--surface)" stroke-width="2">
          <title>${escape(curve.label)}: ${format(curve.value)}, slope ${format(curve.slope)}</title>
        </circle>`;
    }
    svg += '</g></svg>';
    const readings = series.map(curve => {
      const direction = curve.slope < 0 ? 'A small move right lowers' : curve.slope > 0 ? 'A small move right raises' : 'A small move has no first-order effect on';
      return `<p class="graph-reading" data-slope="${curve.slope}" data-value="${curve.value}"><strong>${escape(curve.label)}:</strong>
        ${escape(yLabel)} = <code>${format(curve.value)}</code>; slope = <code>${format(curve.slope)}</code>.
        ${direction} ${escape(yLabel)} locally, holding everything else fixed.</p>`;
    }).join('');
    return `<figure class="derivative-graph" data-x="${x}">
      <figcaption><h3>${escape(title)}</h3><p>${escape(context)}</p></figcaption>
      <div class="graph-key">${series.map((curve, i) => `<span><i class="${i ? 'comparison' : ''}"></i>${escape(curve.label)}</span>`).join('')}<span><i class="tangent"></i>Dashed: local slope</span></div>
      <p class="graph-hint">Swipe the plot to read all axes. Exact values and slopes are also listed below.</p>
      <div class="graph-plot" tabindex="0" role="region" aria-label="${escape(title)}: scrollable plot">${svg}</div>
      <p class="graph-reading">Selected ${escape(xLabel)}: <strong>${format(x)}</strong>. The dot is the current value, not a new training update.</p>${readings}
    </figure>`;
  }

  return { slope };
})();
