import { batchSchedule, kvProjectionWork, rooflinePoint, weightMemoryGB } from '../lib/book-models';

const book = document.querySelector<HTMLElement>('[data-book]');
if (book) {
  const find = <T extends HTMLElement = HTMLElement>(selector: string) => book.querySelector<T>(selector)!;
  const all = <T extends HTMLElement = HTMLElement>(selector: string) => [...book.querySelectorAll<T>(selector)];
  const spreads = all<HTMLElement>('[data-spread]');
  const tabs = all<HTMLAnchorElement>('[data-book-tab]');
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  let current = 0;
  let turn: Animation | undefined;
  let batchTimer: number | undefined;

  // Chapter 01: GPU factory — a teaching model of resource bottlenecks.
  const delivery = find<HTMLInputElement>('#hardware-delivery');
  const compute = find<HTMLInputElement>('#hardware-compute');
  function updateHardware() {
    const d = Number(delivery.value);
    const c = Number(compute.value);
    find<HTMLOutputElement>('#hardware-delivery-value').textContent = `${d} / 3`;
    find<HTMLOutputElement>('#hardware-compute-value').textContent = `${c} / 3`;
    const waiting = d > c ? 5 : d === c ? 1 : 0;
    all<HTMLElement>('[data-hardware-queue] i').forEach((cell, i) => {
      cell.classList.toggle('is-waiting', i < waiting);
    });
    all<HTMLElement>('[data-hardware-workers] i').forEach((cell, i) => {
      cell.classList.toggle('is-idle', d < c && i >= d);
    });
    find('[data-hardware-queue-label]').textContent = d > c ? 'Queue builds' : d < c ? 'No backlog' : 'Short queue';
    find('[data-hardware-worker-label]').textContent = d < c ? 'Some idle' : 'Working';
    find('[data-hardware-roofline-label]').textContent = d < c ? 'Memory-bound' : d > c ? 'Compute-bound' : 'Balanced resources';
    find('[data-hardware-explanation]').textContent = d < c
      ? 'Workers are ready, but deliveries are too slow. Some workers wait for data: the memory-bound side of the roofline.'
      : d > c
        ? 'Materials arrive faster than workers can process them. Pending work builds up: the compute-bound side of the roofline.'
        : 'Delivery and compute are balanced. The workers have enough data to keep moving.';
  }
  delivery.addEventListener('input', updateHardware);
  compute.addEventListener('input', updateHardware);
  updateHardware();

  function stopBatch() {
    window.clearInterval(batchTimer);
    batchTimer = undefined;
    find('[data-batch-play]').textContent = 'Play schedule →';
  }

  function selectPage(index: number, updateHash = true, focus = false) {
    if (index < 0 || index >= spreads.length) return;
    const direction = index >= current ? 1 : -1;
    stopBatch();
    turn?.cancel();
    current = index;
    spreads.forEach((spread, i) => { spread.hidden = i !== current; });
    tabs.forEach((tab, i) => {
      if (i === current) tab.setAttribute('aria-current', 'step');
      else tab.removeAttribute('aria-current');
    });
    find<HTMLButtonElement>('[data-book-prev]').disabled = current === 0;
    find<HTMLButtonElement>('[data-book-next]').disabled = current === spreads.length - 1;
    find('[data-book-position]').textContent = `${current === 0 ? 'Cover' : `Lesson ${current}`} · ${current + 1} / ${spreads.length}`;
    find('[data-book-progress]').style.width = `${(current + 1) / spreads.length * 100}%`;
    if (updateHash) history.replaceState(null, '', `#${spreads[current].dataset.spread}`);
    if (!reducedMotion.matches) {
      turn = spreads[current].animate([
        { opacity: .25, transform: `perspective(1800px) rotateY(${direction * 7}deg) translateX(${direction * 12}px)` },
        { opacity: 1, transform: 'perspective(1800px) rotateY(0) translateX(0)' },
      ], { duration: 430, easing: 'cubic-bezier(.2,.7,.2,1)' });
    }
    if (focus) {
      const heading = spreads[current].querySelector<HTMLElement>('h2, h3')!;
      heading.tabIndex = -1;
      heading.focus({preventScroll: true});
      spreads[current].scrollIntoView({block: 'start', behavior: reducedMotion.matches ? 'instant' : 'smooth'});
    }
  }

  tabs.forEach((tab, i) => tab.addEventListener('click', event => { event.preventDefault(); selectPage(i, true, true); }));
  all<HTMLButtonElement>('[data-book-go]').forEach(button => button.addEventListener('click', () => selectPage(Number(button.dataset.bookGo), true, true)));
  find('[data-book-prev]').addEventListener('click', () => selectPage(current - 1, true, true));
  find('[data-book-next]').addEventListener('click', () => selectPage(current + 1, true, true));
  book.addEventListener('keydown', event => {
    const target = event.target as HTMLElement;
    if (target.closest('input, select, textarea, [contenteditable]') || event.altKey || event.ctrlKey || event.metaKey) return;
    if (event.key === 'ArrowRight' || event.key === 'ArrowLeft') {
      event.preventDefault();
      selectPage(current + (event.key === 'ArrowRight' ? 1 : -1), true, true);
    }
  });
  function readHash() {
    const i = spreads.findIndex(spread => `#${spread.dataset.spread}` === window.location.hash);
    if (i >= 0) selectPage(i, false);
  }
  window.addEventListener('hashchange', readHash);
  book.classList.add('is-ready');
  find('[data-book-navigation]').hidden = false;
  selectPage(0, false);
  readHash();

  // One explicit click per phase or output token; never silently autoplay.
  let tokenStep = -1;
  const outputTokens = all('[data-token]');
  function drawTokens() {
    all('.prompt-token').forEach(token => token.classList.toggle('processed', tokenStep >= 0));
    outputTokens.forEach((token, i) => {
      token.classList.toggle('revealed', i < tokenStep);
      token.setAttribute('aria-hidden', String(i >= tokenStep));
    });
    find<HTMLButtonElement>('[data-token-next]').disabled = tokenStep === outputTokens.length;
    find('[data-token-next]').textContent = tokenStep < 0 ? 'Run prefill →' : tokenStep === outputTokens.length ? 'Reply complete ✓' : 'Generate next token →';
    find('[data-token-phase]').textContent = tokenStep < 0 ? 'Ready to read the prompt' : tokenStep === 0 ? 'Prefill complete · prompt processed' : `Decode · ${tokenStep} / ${outputTokens.length} output tokens`;
    find('[data-token-explanation]').textContent = tokenStep < 0
      ? 'Start with prefill. The three prompt tokens pass through the model together.'
      : tokenStep === 0 ? 'The prompt is processed. Now each click selects the next token and appends it to the context.'
      : tokenStep === outputTokens.length ? 'Done. The six output tokens required six sequential choices. This example treats the final punctuation as its own token.'
      : `Generated “${outputTokens[tokenStep - 1].textContent}”. The next prediction can use the prompt and all ${tokenStep} generated token${tokenStep > 1 ? 's' : ''}.`;
    if (!reducedMotion.matches && tokenStep > 0) outputTokens[tokenStep - 1].animate([{transform:'translateY(8px)',opacity:0},{transform:'translateY(0)',opacity:1}], {duration:300});
  }
  find('[data-token-next]').addEventListener('click', () => { tokenStep++; drawTokens(); });
  find('[data-token-reset]').addEventListener('click', () => { tokenStep = -1; drawTokens(); });

  const cacheLength = find<HTMLInputElement>('#cache-length');
  function drawCache() {
    const prompt = Number(cacheLength.value);
    find('#cache-length-value').textContent = `${prompt} tokens`;
    const work = kvProjectionWork(prompt, 3);
    for (const cached of [false, true]) {
      const container = find(cached ? '[data-cache-cached]' : '[data-cache-uncached]');
      container.replaceChildren();
      for (let step = 0; step < 3; step++) {
        const row = document.createElement('div');
        const label = document.createElement('small'); label.textContent = `${step + 1}`; row.append(label);
        for (let pos = 0; pos < prompt + step + 1; pos++) {
          const square = document.createElement('i');
          square.className = cached && pos < prompt + step ? 'cache-saved' : 'cache-new';
          square.style.setProperty('--delay', `${step * 90 + pos * 8}ms`);
          row.append(square);
        }
        container.append(row);
      }
    }
    find('[data-cache-uncached-count]').textContent = String(work.uncached);
    find('[data-cache-explanation]').textContent = `After the shared prefill, these three steps project ${work.uncached} token positions without caching, versus ${work.cached} new positions with caching. Earlier K/V entries are reused, but attention still reads them.`;
  }
  cacheLength.addEventListener('input', drawCache);
  drawCache();

  const intensityInput = find<HTMLInputElement>('#roof-intensity');
  function drawRoofline() {
    const intensity = Number(intensityInput.value);
    const point = rooflinePoint(intensity, 1000, 100);
    const x = 55 + intensity / 250 * 370;
    const y = 200 - point.attainable / 100 * 150;
    find('#roof-intensity-value').textContent = `${intensity} FLOPs/byte`;
    find('[data-roof-point]').setAttribute('cx', String(x));
    find('[data-roof-point]').setAttribute('cy', String(y));
    find('[data-roof-guide]').setAttribute('d', `M${x} 200V${y}H55`);
    find('[data-roof-bound]').textContent = intensity === point.ridge ? 'At the ridge point' : point.bound;
    find('[data-roof-ceiling]').textContent = String(point.attainable);
    find('[data-roof-explanation]').textContent = intensity < point.ridge
      ? `At ${intensity} FLOPs per byte, memory can feed only ${point.attainable} TFLOP/s of work. More arithmetic capacity alone cannot raise this limit.`
      : `The memory ceiling is ${intensity} TFLOP/s, but compute caps this GPU at 100 TFLOP/s. The two limits meet at ${point.ridge} FLOPs per byte.`;
  }
  intensityInput.addEventListener('input', drawRoofline);
  drawRoofline();

  all<HTMLInputElement>('input[name="weight-precision"]').forEach(input => input.addEventListener('change', drawWeights));
  function drawWeights() {
    const selected = find<HTMLInputElement>('input[name="weight-precision"]:checked');
    const bits = Number(selected.value);
    const gb = weightMemoryGB(7, bits);
    find('[data-weight-gb]').textContent = String(gb);
    find('[data-weight-fill]').style.width = `${gb / 28 * 100}%`;
    const label = selected.parentElement!.querySelector('span')!.firstChild!.textContent;
    find('[data-weight-explanation]').textContent = `${label} uses ${gb} GB for raw weights: 7 billion × ${bits} bits ÷ 8. ${bits < 32 ? `That is ${32 / bits}× smaller than the FP32 baseline.` : 'This is the 28 GB baseline.'} Quality and runtime overhead are not represented here.`;
  }
  drawWeights();

  const lengths = [2, 5, 3, 2, 4, 1];
  const fixed = batchSchedule(lengths, 3, false);
  const continuous = batchSchedule(lengths, 3, true);
  const maxSteps = fixed.length;
  const stepInput = find<HTMLInputElement>('#batch-step');
  stepInput.max = String(maxSteps);
  function drawSchedule(containerSelector: string, rows: ReturnType<typeof batchSchedule>) {
    const container = find(containerSelector);
    const step = Number(stepInput.value);
    container.replaceChildren();
    for (let time = 0; time < maxSteps; time++) {
      const col = document.createElement('div');
      col.className = `schedule-column${time < step ? ' elapsed' : ''}${time === step - 1 ? ' current' : ''}`;
      col.setAttribute('aria-label', `Step ${time + 1}: ${rows[time]?.filter(r => r !== null).map(r => String.fromCharCode(65 + r.request)).join(', ') || 'complete'}`);
      for (let slot = 0; slot < 3; slot++) {
        const cell = document.createElement('span');
        const request = rows[time]?.[slot];
        if (request) {cell.className = `request-${request.request}`; cell.textContent = String.fromCharCode(65 + request.request);}
        else cell.className = 'empty-slot';
        col.append(cell);
      }
      container.append(col);
    }
  }
  function drawBatch() {
    const step = Number(stepInput.value);
    drawSchedule('[data-fixed-schedule]', fixed);
    drawSchedule('[data-continuous-schedule]', continuous);
    find('#batch-step-value').textContent = `${step} / ${maxSteps}`;
    find('[data-fixed-count]').textContent = `${fixed.length} steps`;
    find('[data-continuous-count]').textContent = `${continuous.length} steps`;
    find('[data-batch-explanation]').textContent = step === 0
      ? 'Press play or scrub the timeline. Each column is one step, each colored cell is one request generating a token.'
      : step >= continuous.length
        ? `Continuous batching completes all 17 token steps in ${continuous.length} rounds; fixed batches need ${fixed.length}. This is a toy scheduling comparison, not a benchmark.`
        : `Step ${step}: fixed batches have generated ${fixed.slice(0,step).flat().filter(Boolean).length} tokens; continuous batching has generated ${continuous.slice(0,step).flat().filter(Boolean).length}. Free slots can admit waiting requests at the next step.`;
  }
  stepInput.addEventListener('input', () => {stopBatch(); drawBatch();});
  find('[data-batch-play]').addEventListener('click', () => {
    if (batchTimer) {stopBatch(); return;}
    if (Number(stepInput.value) >= maxSteps) stepInput.value = '0';
    if (reducedMotion.matches) {
      stepInput.value = String(Math.min(Number(stepInput.value) + 1, maxSteps)); drawBatch(); return;
    }
    find('[data-batch-play]').textContent = 'Pause schedule';
    batchTimer = window.setInterval(() => {
      stepInput.value = String(Number(stepInput.value) + 1);
      drawBatch();
      if (Number(stepInput.value) >= maxSteps) stopBatch();
    }, 650);
  });
  find('[data-batch-reset]').addEventListener('click', () => {stopBatch(); stepInput.value = '0'; drawBatch();});
  document.addEventListener('visibilitychange', () => {if(document.hidden) stopBatch();});
  reducedMotion.addEventListener('change', () => {stopBatch(); turn?.cancel();});
  drawBatch();
}
