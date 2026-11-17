import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import { batchSchedule, kvProjectionWork, rooflinePoint, weightMemoryGB } from '../src/lib/book-models.ts';

assert.equal(weightMemoryGB(7, 32), 28);
assert.equal(weightMemoryGB(7, 16), 14);
assert.equal(weightMemoryGB(7, 8), 7);
assert.equal(weightMemoryGB(7, 4), 3.5);
assert.deepEqual(kvProjectionWork(8, 3), {uncached: 30, cached: 3});
assert.deepEqual(kvProjectionWork(24, 3), {uncached: 78, cached: 3});
assert.deepEqual(rooflinePoint(20, 1000, 100), {attainable:20, ridge:100, bound:'Memory-bound'});
assert.equal(rooflinePoint(100, 1000, 100).attainable, 100);
assert.equal(rooflinePoint(250, 1000, 100).attainable, 100);
assert.equal(rooflinePoint(250, 1000, 100).bound, 'Compute-bound');
const requests = [2,5,3,2,4,1];
for (const continuous of [false, true]) {
  const schedule = batchSchedule(requests, 3, continuous);
  assert.equal(schedule.length, continuous ? 7 : 9);
  assert.equal(schedule.flat().filter(Boolean).length, 17);
  for (let request = 0; request < requests.length; request++) {
    assert.deepEqual(schedule.flat().filter(r => r && r.request === request).map(r => r.step),
      Array.from({length:requests[request]}, (_,i)=>i));
  }
  assert(schedule.every(row => row.length === 3));
  if (!continuous) {
    assert.equal(schedule[2][0], null, 'Finished request A must leave its original slot empty');
    assert.equal(schedule[2][1].request, 1, 'B stays in its original slot');
  } else {
    assert.equal(schedule[2][0].request, 3, 'D fills the slot vacated by A');
    assert.equal(schedule[5][2].request, 4, 'E keeps its slot after its neighbors finish');
  }
}
const html = await fs.readFile(new URL('../dist/concepts/llm-inference-v2/index.html', import.meta.url),'utf8');
assert.equal((html.match(/data-spread=/g) || []).length, 24);
for (const link of html.matchAll(/href="(\/[^"#?]*)/g)) {
  const target = new URL(`../dist${link[1]}${link[1].endsWith('/') ? '' : '/'}index.html`, import.meta.url);
  if (!link[1].match(/\.[a-z]+$/)) await fs.access(target);
}
console.log('Book checks passed: precision storage, KV work, roofline limits, scheduling conservation, twenty-four spreads, and built internal links.');
