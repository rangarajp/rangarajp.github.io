import type { CollectionEntry } from 'astro:content';

export type ConceptEntry = CollectionEntry<'concepts'>;

export type ConceptSeries = {
	id: string;
	title: string;
	description: string;
	hubPath: string;
	seriesOrder: number;
	articles: ConceptEntry[];
};

export function getConceptPath(id: string): string {
	if (id.endsWith('/index')) {
		return `/concepts/${id.slice(0, -'/index'.length)}`;
	}
	return `/concepts/${id}`;
}

/** Series id for an entry. Hub files may be `series` or `series/index`. */
export function getSeriesId(id: string): string | null {
	if (!id.includes('/')) {
		// Folder index.md → id "llm-inference" (Astro glob), or a true standalone
		return id;
	}
	if (id.endsWith('/index')) {
		return id.slice(0, -'/index'.length);
	}
	return id.slice(0, id.indexOf('/'));
}

export function isSeriesHub(id: string): boolean {
	return !id.includes('/') || id.endsWith('/index');
}

export function groupConceptsBySeries(concepts: ConceptEntry[]): {
	series: ConceptSeries[];
	standalone: ConceptEntry[];
} {
	const seriesMap = new Map<string, { hub: ConceptEntry | null; articles: ConceptEntry[] }>();

	for (const entry of concepts) {
		const seriesId = getSeriesId(entry.id);
		if (!seriesId) continue;

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

	const standalone: ConceptEntry[] = [];
	const series: ConceptSeries[] = [];

	for (const [id, group] of seriesMap.entries()) {
		const articles = group.articles.sort(
			(a, b) => (a.data.order ?? Number.MAX_SAFE_INTEGER) - (b.data.order ?? Number.MAX_SAFE_INTEGER),
		);

		// Hub with no chapters = standalone topic at concepts root
		if (articles.length === 0) {
			if (group.hub) standalone.push(group.hub);
			continue;
		}

		const hubTitle = group.hub?.data.title ?? formatSeriesTitle(id);
		const title = hubTitle.replace(/\s+overview$/i, '');

		series.push({
			id,
			title,
			description: group.hub?.data.description ?? '',
			hubPath: getConceptPath(group.hub?.id ?? id),
			seriesOrder: group.hub?.data.seriesOrder ?? Number.MAX_SAFE_INTEGER,
			articles,
		});
	}

	series.sort((a, b) => {
		if (a.seriesOrder !== b.seriesOrder) return a.seriesOrder - b.seriesOrder;
		return a.title.localeCompare(b.title);
	});

	standalone.sort((a, b) => a.data.title.localeCompare(b.data.title));

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
