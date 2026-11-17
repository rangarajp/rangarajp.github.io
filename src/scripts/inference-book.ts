import { batchSchedule, kvProjectionWork, rooflinePoint, weightMemoryGB } from '../lib/book-models';

const book = document.querySelector<HTMLElement>('[data-book]');
if (book) {
  const find = <T extends HTMLElement = HTMLElement>(selector: string) => book.querySelector<T>(selector)!;
  const all = <T extends HTMLElement = HTMLElement>(selector: string) => [...book.querySelectorAll<T>(selector)];
  const spreads = all<HTMLElement>('[data-spread]');
  const tabs = all<HTMLAnchorElement>('[data-book-tab]');
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  const sectionNames = ['Cover','Know your GPU','Meet your hardware','Move the bytes','Time the journey','Count the work','Multiply the numbers','Reuse the weights','Find the speed limit','Trust the measurements','Meet Qwen','Read the prompt','One token at a time','Remember the work','Share the weights','Big matrices, thin vectors','Reuse changes the limit','Look inside the GPU','The cost of a reply','One token at a time','Remember the work','Find the bottleneck','Make room for more','Keep the GPU busy'];
  let current = 0;
  let turn: Animation | undefined;
  let batchTimer: number | undefined;


  // Chapter 02: each interactive figure is a small teaching model or a labeled measurement.
  const q2 = (selector: string) => find<HTMLElement>(selector);
  const tokenExamples = [['The','GPU','reads','tokens'],['One','token','at','a','time'],['Cache','keys','and','values']];
  const tokenSelect = find<HTMLSelectElement>('#q2-token-input');
  function drawQ2Tokens() {
    q2('[data-q2-tokens]').innerHTML = tokenExamples[Number(tokenSelect.value)].map((w,i)=>'<span class="q2-cell" style="animation-delay:'+i*90+'ms">'+w+'</span>').join('');
  }
  tokenSelect.addEventListener('change',drawQ2Tokens); drawQ2Tokens();
  const prefillSlider = find<HTMLInputElement>('#q2-prefill-size');
  function drawQ2Prefill(active=false) {
    const n=[8,16,24,32][Number(prefillSlider.value)-1];
    q2('#q2-prefill-value').textContent=String([32,64,96,128][Number(prefillSlider.value)-1]);
    q2('[data-q2-prefill-grid]').innerHTML=Array.from({length:n},(_,i)=>'<span class="q2-cell '+(active?'is-on':'')+'" style="animation-delay:'+i*16+'ms"></span>').join('');
  }
  prefillSlider.addEventListener('input',()=>drawQ2Prefill()); find('[data-q2-prefill-play]').addEventListener('click',()=>drawQ2Prefill(true));drawQ2Prefill();
  let decodeStep=0;
  function drawQ2Decode(){
    q2('[data-q2-decode-grid]').innerHTML=['Prompt',...['The','GPU','works','fast'].slice(0,decodeStep)].map((t,i)=>'<span class="q2-cell '+(i?'is-on':'')+'">'+t+'</span>').join('');
    q2('[data-q2-decode-count]').textContent=decodeStep+' / 4';
  }
  find('[data-q2-decode-next]').addEventListener('click',()=>{decodeStep=decodeStep===4?0:decodeStep+1;drawQ2Decode();});
  find('[data-q2-decode-reset]').addEventListener('click',()=>{decodeStep=0;drawQ2Decode();});drawQ2Decode();
  const cacheData=[{n:32,on:3.80609,off:4.42215},{n:128,on:3.84876,off:5.84706},{n:512,on:4.07042,off:12.39456}];
  const cacheSlider=find<HTMLInputElement>('#q2-cache-context');
  function drawQ2Cache(){
    const d=cacheData[Number(cacheSlider.value)];
    q2('#q2-cache-context-value').textContent=String(d.n);
    q2('[data-q2-cache-on-time]').textContent=d.on.toFixed(2)+' s';
    q2('[data-q2-cache-off-time]').textContent=d.off.toFixed(2)+' s';
    q2('[data-q2-cache-on]').style.width=(d.on/12.39456*100)+'%';
    q2('[data-q2-cache-off]').style.width=(d.off/12.39456*100)+'%';
    q2('[data-q2-cache-speed]').textContent=(d.off/d.on).toFixed(2)+'×';
  }
  cacheSlider.addEventListener('input',drawQ2Cache);drawQ2Cache();
  const batchData=[{b:1,t:8.27185,ms:3868.54},{b:2,t:16.05551,ms:3986.17},{b:4,t:30.78599,ms:4157.74}];
  const batchSlider=find<HTMLInputElement>('#q2-batch-size');
  function drawQ2Batch(){
    const d=batchData[Number(batchSlider.value)];
    q2('#q2-batch-value').textContent=String(d.b);
    q2('[data-q2-batch-rows]').innerHTML=Array.from({length:d.b},(_,i)=>'<div class="q2-batch-row"><span>REQUEST '+(i+1)+'</span><i></i><i></i><i></i><i></i><i></i></div>').join('');
    q2('[data-q2-batch-throughput]').textContent=d.t.toFixed(2)+' tok/s';
    q2('[data-q2-batch-latency]').textContent=(d.ms/1000).toFixed(2)+' s';
  }
  batchSlider.addEventListener('input',drawQ2Batch);drawQ2Batch();
  let isDecodeMatrix=false;
  function drawQ2Matrix(){
    q2('[data-q2-matrix-input]').innerHTML=Array.from({length:isDecodeMatrix?3:9},()=>'<i></i>').join('');
    q2('[data-q2-matrix-output]').innerHTML=Array.from({length:isDecodeMatrix?3:9},()=>'<i></i>').join('');
    q2('[data-q2-matrix-label]').textContent=isDecodeMatrix?'DECODE · ONE ROW × WEIGHTS':'PREFILL · MATRIX × MATRIX';
    q2('[data-q2-matrix-input]').classList.toggle('is-thin',isDecodeMatrix);
    q2('[data-q2-matrix-output]').classList.toggle('is-thin',isDecodeMatrix);
    find('[data-q2-matrix-toggle]').textContent=isDecodeMatrix?'Switch to prefill →':'Switch to decode →';
  }
  find('[data-q2-matrix-toggle]').addEventListener('click',()=>{isDecodeMatrix=!isDecodeMatrix;drawQ2Matrix();});drawQ2Matrix();
  const aiSlider=find<HTMLInputElement>('#q2-ai-batch');
  function drawQ2AI(){
    const b=Number(aiSlider.value);
    q2('#q2-ai-batch-value').textContent=String(b);
    q2('[data-q2-ai]').textContent=b+' FLOP'+(b===1?'':'s')+'/byte';
    q2('[data-q2-ai-marker]').style.left=(b/4*100)+'%';
  }
  aiSlider.addEventListener('input',drawQ2AI);drawQ2AI();
  const profileMode=find<HTMLSelectElement>('#q2-profile-mode');
  function drawQ2Profile(){
    const dec=profileMode.value==='decode', mm=dec?293.13:117.24, att=dec?1.37:0;
    q2('[data-q2-profile-mm-value]').textContent=mm.toFixed(2)+' ms';
    q2('[data-q2-profile-attn-value]').textContent=dec?'1.37 ms':'Not listed';
    q2('[data-q2-profile-mm]').style.width=(mm/293.13*100)+'%';
    q2('[data-q2-profile-attn]').style.width=(att/293.13*100)+'%';
    q2('[data-q2-profile-calls]').textContent=dec?'339 across 3 steps':'113 in 1 pass';
    q2('[data-q2-profile-note]').textContent=dec?'The decode capture covers three forwards. The 293.13 ms matrix-multiply total averages 97.71 ms per forward. The efficient-attention operator took 1.37 ms across 84 calls.':'One prompt forward accounts for the 117.24 ms matrix-multiply total. The attention timing is unlisted in the prefill operator summary.';
  }
  profileMode.addEventListener('change',drawQ2Profile);drawQ2Profile();
  const caseChoice=find<HTMLSelectElement>('#q2-case-choice');
  const caseExplanations = [
    'The 2.88 GiB of FP16 weights occupy almost all of the device memory PyTorch allocates when this Qwen model is loaded.',
    'For one 512-token prompt and 32 new tokens, caching made generation 3.05 times as fast. Each decode step reused the keys and values from the earlier positions.',
    'A batch of four requests produced 30.79 output tokens per second in total. Each request began with a 128-token prompt and produced 32 tokens.',
    'With a 128-token prompt cache ready, one batch-one decode forward took 117.26 ms. This pass uses the previous token as input and produces scores for the next token.',
  ];
  function drawQ2Case(){q2('[data-q2-case-explain]').textContent=caseExplanations[Number(caseChoice.value)];}
  caseChoice.addEventListener('change',drawQ2Case);drawQ2Case();

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
    find('[data-hardware-explanation]').textContent = d < c
      ? 'Workers are ready, but deliveries are too slow. Some workers wait for data.'
      : d > c
        ? 'Materials arrive faster than workers can process them. Pending work builds up.'
        : 'Delivery and compute are balanced. The workers have enough data to keep moving.';
  }
  delivery.addEventListener('input', updateHardware);
  compute.addEventListener('input', updateHardware);
  updateHardware();


  // Hardware chapter: five more interactive spreads, in the same book runtime.
  const copyCount = find<HTMLInputElement>('#copy-count');
  function updateCopy() {
    const n = Number(copyCount.value);
    const mb = n * 2;
    find('#copy-count-value').textContent = n + ' million';
    find('[data-copy-read]').textContent = mb + ' MB';
    find('[data-copy-write]').textContent = mb + ' MB';
    find('[data-copy-total]').textContent = (mb * 2) + ' MB';
    find('[data-copy-time]').textContent = (mb * 2 / 237.5).toFixed(3) + ' ms';
  }
  copyCount.addEventListener('input', updateCopy); updateCopy();

  let matrixRow = 0;
  const matrixCells = all<HTMLElement>('[data-matrix-a] span');
  const matrixResults = all<HTMLElement>('[data-matrix-result] span');
  function resetMatrix() {
    matrixRow = 0;
    matrixCells.forEach(c => c.classList.remove('is-active'));
    matrixResults.forEach(c => { c.textContent = '?'; c.classList.remove('is-done'); });
    find('[data-matrix-explanation]').textContent = 'Start with the first row. Each output is the sum of three multiplications.';
  }
  find('[data-matrix-next]').addEventListener('click', () => {
    if (matrixRow >= 2) resetMatrix();
    matrixCells.forEach((c, i) => c.classList.toggle('is-active', Math.floor(i / 3) === matrixRow));
    matrixResults[matrixRow].textContent = matrixRow === 0 ? '16' : '37';
    matrixResults[matrixRow].classList.add('is-done');
    find('[data-matrix-explanation]').textContent = matrixRow === 0
      ? 'First row: (1 × 2) + (2 × 1) + (3 × 4) = 16.'
      : 'Second row: (4 × 2) + (5 × 1) + (6 × 4) = 37.';
    matrixRow++;
  });
  find('[data-matrix-reset]').addEventListener('click', resetMatrix);

  const reuse = [
    {b:1,ai:1,tf:0.15,estimated:true},{b:32,ai:31.51,tf:6.32,estimated:true},
    {b:128,ai:120.47,tf:20.689},{b:512,ai:409.6,tf:43.733},
    {b:1024,ai:682.67,tf:34.632},{b:2048,ai:1024,tf:36.253},
    {b:4096,ai:1365.33,tf:38.289}
  ];
  const reuseInput = find<HTMLInputElement>('#reuse-batch');
  function updateReuse() {
    const d = reuse[Number(reuseInput.value)];
    find('#reuse-batch-value').textContent = String(d.b);
    find('[data-reuse-ai]').textContent = d.ai.toFixed(1);
    find('[data-reuse-tf]').textContent = d.tf.toFixed(2) + ' TFLOP/s';
    find<HTMLElement>('[data-reuse-ai-bar]').style.width = (d.ai / 1365.33 * 100) + '%';
    find<HTMLElement>('[data-reuse-tf-bar]').style.width = (d.tf / 43.733 * 100) + '%';
    find('[data-reuse-explanation]').textContent = d.estimated
      ? 'This small-batch point is an estimate because the original trials varied substantially. More reuse generally raises arithmetic intensity.'
      : d.b === 512
        ? 'At 512 columns, the weights can be reused across many inputs. This was the fastest throughput measured in our sweep.'
        : 'This point was measured in the notebook. Higher batch size does not always mean higher throughput.';
  }
  reuseInput.addEventListener('input', updateReuse); updateReuse();

  const roofInput = find<HTMLInputElement>('#hardware-roof-slider');
  function updateRoof() {
    const ai = Number(roofInput.value);
    const bound = ai < 184.13;
    find('#hardware-roof-value').textContent = ai + ' FLOPs/byte';
    find('[data-hardware-roof-description]').textContent = bound
      ? 'Memory-bound: workers can calculate, but the next bytes have not arrived. Reuse data or reduce transfers.'
      : 'Compute-bound: the workers have enough data to stay busy. Faster arithmetic and efficient matrix kernels may help.';
    const x = 40 + Math.min(ai / 400, 1) * 385;
    const y = 207 - Math.min(ai / 184.13, 1) * 152;
    const cursor = book.querySelector<SVGCircleElement>('[data-roofline-cursor]')!;
    cursor.setAttribute('cx', String(x)); cursor.setAttribute('cy', String(y));
  }
  roofInput.addEventListener('input', updateRoof); updateRoof();

  all<HTMLButtonElement>('[data-evidence-answer]').forEach(button => button.addEventListener('click', () => {
    const answer = button.dataset.evidenceAnswer;
    find('[data-evidence-feedback]').textContent = answer === 'neither'
      ? 'Correct. Repeat under controlled conditions. The median summarizes the run; the fastest trial suggests a possible path, but neither alone proves stable speed.'
      : answer === 'median'
        ? 'The median summarizes the trials, but these trials appear to have mixed operating conditions. It does not establish a stable baseline.'
        : 'The fastest trial is promising, but a single fast result does not establish repeatability. We need controlled repeated runs.';
  }));


  // Additional hardware spreads: device inspection, copy latency, and FLOP/s.
  const partDetails: Record<string,string> = {
    vram:'VRAM stores model weights, activations and caches. Our GPU has 6 GiB of device memory. A model must fit alongside its working tensors.',
    sm:'Streaming multiprocessors (SMs) execute many threads. Our RTX A3000 laptop GPU has 32 SMs. More SMs do not automatically mean a faster workload.',
    clock:'GPU clocks vary with power, temperature and workload. We observed low idle clocks and much higher clocks under sustained GEMM load.',
    tensor:'Tensor Cores accelerate supported matrix operations. Their usefulness depends on data types, shapes, and which kernel is selected.'
  };
  all<HTMLButtonElement>('[data-part]').forEach(button=>button.addEventListener('click',()=>{
    all<HTMLButtonElement>('[data-part]').forEach(b=>b.setAttribute('aria-pressed',String(b===button)));
    find('[data-part-explanation]').textContent=partDetails[button.dataset.part || 'vram'];
    find('[data-inspect-vram]').classList.toggle('highlight',button.dataset.part==='vram');
    find('[data-inspect-sms]').classList.toggle('highlight',button.dataset.part!=='vram');
  }));
  const bwInput=find<HTMLInputElement>('#latency-bandwidth');
  function updateLatency(){
    const bw=Number(bwInput.value),ms=64/bw;
    find('#latency-bandwidth-value').textContent=bw.toFixed(1)+' GB/s';
    find('[data-latency-time]').textContent=ms.toFixed(3)+' ms';
    find('[data-latency-rate]').textContent=bw.toFixed(1)+' GB/s';
    find('[data-latency-us]').textContent=Math.round(ms*1000)+' μs';
    find<HTMLElement>('[data-latency-bar]').style.width=Math.min(100,25/bw*100)+'%';
    const track=find<HTMLElement>('.hardware-delivery-demo .hardware-track');
    track.style.setProperty('--delivery-duration',(2.4*237.5/bw).toFixed(2)+'s');
  }
  bwInput.addEventListener('input',updateLatency);updateLatency();
  const flopsData=[
    {n:1,ms:0.224,estimated:true},{n:32,ms:0.170,estimated:true},
    {n:128,ms:0.208},{n:512,ms:0.393},
    {n:1024,ms:0.992},{n:2048,ms:1.896},{n:4096,ms:3.590}
  ];
  const flopsInput=find<HTMLInputElement>('#flops-columns');
  function updateFlops(){
    const d=flopsData[Number(flopsInput.value)];
    const gf=2*4096*4096*d.n/1e9;
    const tf=gf/d.ms/1000;
    find('#flops-columns-value').textContent=String(d.n);
    find('[data-flops-work]').textContent=gf.toFixed(2)+' GFLOPs';
    find('[data-flops-ms]').textContent=d.ms.toFixed(3)+' ms';
    find('[data-flops-rate]').textContent=tf.toFixed(2)+' TFLOP/s';
    find('[data-flops-giga]').textContent=Math.round(tf*1000).toLocaleString('en-US')+' GFLOP/s';
    find<HTMLElement>('[data-flops-blocks]').style.width=Math.max(8,Math.round(d.n/4096*100))+'%';
    find('[data-flops-explanation]').textContent=d.estimated
      ? 'This small-batch timing is an estimate, not a stable measurement. FLOP/s is calculated by dividing the total operations by the elapsed time.'
      : 'This workload requires '+gf.toFixed(2)+' billion operations. Dividing by '+d.ms.toFixed(3)+' ms gives '+tf.toFixed(2)+' trillion operations per second.';
  }
  flopsInput.addEventListener('input',updateFlops);updateFlops();

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
    find('[data-book-position]').textContent = current === 0 ? 'Cover' : current <= 9 ? `Chapter 01 · Lesson ${current} of 9` : current <= 18 ? `Chapter 02 · Lesson ${current - 9} of 9` : sectionNames[current];
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
