// @ts-check

import mdx from '@astrojs/mdx';
import sitemap from '@astrojs/sitemap';
import { defineConfig, fontProviders } from 'astro/config';
import { satteri } from '@astrojs/markdown-satteri';
import { satteriKatex } from 'satteri-katex';

// https://astro.build/config
export default defineConfig({
	site: 'https://rangarajp.github.io',
	integrations: [mdx(), sitemap()],
	markdown: {
		processor: satteri({
			features: { math: true },
			mdastPlugins: [satteriKatex()],
		}),
	},
	fonts: [
		{
			provider: fontProviders.local(),
			name: 'Atkinson',
			cssVariable: '--font-atkinson',
			fallbacks: ['sans-serif'],
			options: {
				variants: [
					{
						src: ['./src/assets/fonts/atkinson-regular.woff'],
						weight: 400,
						style: 'normal',
						display: 'swap',
					},
					{
						src: ['./src/assets/fonts/atkinson-bold.woff'],
						weight: 700,
						style: 'normal',
						display: 'swap',
					},
				],
			},
		},
	],
});
