import type { CollectionEntry } from 'astro:content';

export type ConceptEntry = CollectionEntry<'concepts'>;

export type ConceptSeries = {
	id: string;
	title: string;
	description: string;
	hubPath: string;
	articles: ConceptEntry[];
};

export function getConceptPath(id: string): string {
	return id.endsWith('/index') ? `/concepts/${id.slice(0, -'/index'.length)}` : `/concepts/${id}`;
}

export function getSeriesId(id: string): string | null {
	const slashIndex = id.indexOf('/');
	return slashIndex === -1 ? null : id.slice(0, slashIndex);
}

export function isSeriesHub(id: string): boolean {
	return id.endsWith('/index');
}

export function groupConceptsBySeries(concepts: ConceptEntry[]): {
	series: ConceptSeries[];
	standalone: ConceptEntry[];
} {
	const seriesMap = new Map<string, { hub: ConceptEntry | null; articles: ConceptEntry[] }>();
	const standalone: ConceptEntry[] = [];

	for (const entry of concepts) {
		const seriesId = getSeriesId(entry.id);

		if (!seriesId) {
			standalone.push(entry);
			continue;
		}

		if (!seriesMap.has(seriesId)) {
			seriesMap.set(seriesId, { hub: null, articles: [] });
		}

		const group = seriesMap.get(seriesId)!;

		if (isSeriesHub(entry.id)) {
			group.hub = entry;
		} else {
			group.articles.push(entry);
		}
	}

	const series = [...seriesMap.entries()]
		.map(([id, group]) => {
			const articles = group.articles.sort(
				(a, b) => (a.data.order ?? Number.MAX_SAFE_INTEGER) - (b.data.order ?? Number.MAX_SAFE_INTEGER),
			);

			const hubTitle = group.hub?.data.title ?? formatSeriesTitle(id);
			const title = hubTitle.replace(/\s+overview$/i, '');

			return {
				id,
				title,
				description: group.hub?.data.description ?? '',
				hubPath: getConceptPath(group.hub?.id ?? `${id}/index`),
				articles,
			};
		})
		.sort((a, b) => a.title.localeCompare(b.title));

	return { series, standalone };
}

export function getSeriesContext(
	entry: ConceptEntry,
	concepts: ConceptEntry[],
): { series: ConceptSeries; currentIndex: number } | null {
	const seriesId = getSeriesId(entry.id);
	if (!seriesId) return null;

	const { series } = groupConceptsBySeries(concepts);
	const match = series.find((item) => item.id === seriesId);
	if (!match) return null;

	const currentIndex = match.articles.findIndex((article) => article.id === entry.id);

	return { series: match, currentIndex };
}

function formatSeriesTitle(id: string): string {
	return id
		.split('-')
		.map((part) => part.charAt(0).toUpperCase() + part.slice(1))
		.join(' ');
}
