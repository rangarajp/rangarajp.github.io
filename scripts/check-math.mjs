import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import { createMarkdownProcessor } from '@astrojs/markdown-remark';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

const root = new URL('../', import.meta.url);
const processor = await createMarkdownProcessor({
 remarkPlugins: [remarkMath], rehypePlugins: [rehypeKatex],
});
let files = 0, equations = 0;
const excluded = 'llm_inference_engineering_concepts_summary_fixed.md'; // excluded by the content collection
for (const entry of await fs.readdir(new URL('src/content/', root), { recursive: true })) {
 if (!/\.mdx?$/.test(entry) || path.basename(entry) === excluded) continue;
 const source = await fs.readFile(new URL(`src/content/${entry}`, root), 'utf8');
 const prose = source.replace(/^---\n[\s\S]*?\n---\n/, '').replace(/^\s*(```|~~~)[\s\S]*?^\s*\1[^\n]*$/gm, '').replace(/`[^`]*`/g, '');
 assert(!/\\\[|\\\]|\\\(|\\\)/.test(prose), `Unsupported math delimiters: ${entry}`);
 const rendered = await processor.render(source.replace(/^---\n[\s\S]*?\n---\n/, ''));
 assert(!rendered.code.includes('katex-error'), `Invalid KaTeX in ${entry}`);
 equations += (rendered.code.match(/class="katex"/g) || []).length;
 files++;
}
// Render real article bodies as regression cases: aligned RoPE and Q/K/V projections.
const rope = (await processor.render(await fs.readFile(new URL('src/content/concepts/llm-inference-v2/learn/02-transformer-forward-pass.md', root), 'utf8'))).code;
assert(rope.includes('katex-display') && rope.includes('aligned'), 'RoPE must render as aligned display math');
const qkv = (await processor.render(await fs.readFile(new URL('src/content/concepts/llm-inference-v2/learn/02-transformer-forward-pass.md', root), 'utf8'))).code;
assert(qkv.includes('katex-display') && !qkv.includes('katex-error'), 'Q/K/V projections must render');
for (const entry of await fs.readdir(new URL('dist/', root), { recursive: true })) {
 if (!entry.endsWith('.html')) continue;
 const html = await fs.readFile(new URL(`dist/${entry}`, root), 'utf8');
 assert(!html.includes('class="katex-error"'), `Rendered KaTeX error: ${entry}`);
}
console.log(`Math checks passed: ${files} published Markdown files, ${equations} rendered equations; RoPE and Q/K/V regression cases pass.`);
