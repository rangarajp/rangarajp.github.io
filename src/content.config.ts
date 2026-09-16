import { defineCollection } from 'astro:content';
import { glob } from 'astro/loaders';
import { z } from 'astro/zod';

const concepts = defineCollection({
	loader: glob({ base: './src/content/concepts', pattern: '**/*.{md,mdx}' }),
	schema: ({ image }) =>
		z.object({
			title: z.string(),
			description: z.string(),
			pubDate: z.coerce.date(),
			updatedDate: z.coerce.date().optional(),
			heroImage: z.optional(image()),
			/** Chapter order within a series (articles only). */
			order: z.number().optional(),
			/** Book order on /concepts (hub pages only). */
			seriesOrder: z.number().optional(),
		}),
});

export const collections = { concepts };
