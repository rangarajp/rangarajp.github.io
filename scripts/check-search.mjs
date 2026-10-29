import fs from 'node:fs/promises';
import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';

// Exercise the generated browser index using local files, without a web server.
const bundle = new URL('../dist/pagefind/pagefind.js', import.meta.url);
const originalFetch = globalThis.fetch;
globalThis.fetch = async (url) => {
 const file = fileURLToPath(new URL(String(url).split('?')[0]));
 const bytes = await fs.readFile(file);
 return new Response(bytes, {
  headers: { 'Content-Type': file.endsWith('.wasm') ? 'application/wasm' : 'application/octet-stream' },
 });
};
let instance;
try {
 const pagefind = await import(bundle.href);
 instance = pagefind.createInstance({ basePath: new URL('./', bundle).href, baseUrl: '/' });
 const filters = await instance.filters();
 for (const section of ['Inference original', 'Inference v2', 'Build from Scratch']) {
  assert(filters.Section[section] > 0, `Missing section: ${section}`);
 }
 for (const section of ['Inference original', 'Inference v2']) {
  const response = await instance.search('quantization', { filters: { Section: section } });
  assert(response.results.length > 0, `No quantization results in ${section}`);
  const rows = await Promise.all(response.results.map(hit => hit.data()));
  assert(rows.every(row => row.filters.Section.includes(section)));
  for (const row of rows) {
   await fs.access(new URL(`../dist${row.url.split('#')[0]}index.html`, import.meta.url));
  }
 }
 const body = await instance.search('"copy-on-write"');
 assert(body.results.length > 0, 'Article body must be searchable');
 const none = await instance.search('"zzzxxyyqqwweerrtt"');
 assert.equal(none.results.length, 0, 'Exact nonexistent phrase should have no results');
 const planned = await instance.search('"This chapter is planned"');
 assert.equal(planned.results.length, 0, 'Planned chapter placeholders should not be indexed');
 console.log('Search checks passed: both guides, section filters, valid URLs, body search, empty results, and placeholder exclusion.');
} finally {
 if (instance) await instance.destroy();
 globalThis.fetch = originalFetch;
}
