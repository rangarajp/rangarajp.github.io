import type { CollectionEntry } from 'astro:content';

export const INFERENCE_V2_BASE = '/concepts/llm-inference-v2';

export type ChapterStatus = 'planned' | 'draft' | 'in-progress' | 'complete';
export type Difficulty = 'intro' | 'intermediate' | 'advanced';

export type NavPage = {
	slug: string;
	title: string;
	description: string;
	section: 'home' | 'learn' | 'reference' | 'journal';
	part?: number;
	chapter?: number;
	status: ChapterStatus;
	difficulty?: Difficulty;
	tags?: string[];
};

export const PART_TITLES: Record<number, string> = {
	1: 'Inside LLM inference',
	2: 'Performance fundamentals',
	3: 'Making inference faster',
	4: 'Scaling beyond one GPU',
	5: 'Building inference engines',
	6: 'Production LLM serving',
};

export const NAV_PAGES: NavPage[] = [
	{
		slug: '',
		title: 'Inference Engineering Handbook',
		description: 'Understand, derive, implement, measure, optimize, and operate LLM inference.',
		section: 'home',
		status: 'complete',
	},
	{
		slug: 'learn',
		title: 'Learn',
		description: 'Six-part curriculum. Theory and practical work live in the same chapter.',
		section: 'learn',
		status: 'in-progress',
	},
	{
		slug: 'learn/from-text-to-tokens',
		title: 'From text to tokens',
		description: 'Token IDs, embedding rows, and how position is applied at inference time.',
		section: 'learn',
		part: 1,
		chapter: 1,
		status: 'in-progress',
		difficulty: 'intro',
		tags: ['tokenization', 'rope'],
	},
	{
		slug: 'learn/transformer-forward-pass',
		title: 'One forward pass',
		description: 'Causal attention, the feed-forward network, residuals, and tensor shapes.',
		section: 'learn',
		part: 1,
		chapter: 2,
		status: 'in-progress',
		difficulty: 'intro',
	},
	{
		slug: 'learn/logits-and-sampling',
		title: 'From hidden states to the next token',
		description: 'Logits, softmax, temperature, top-k, and top-p.',
		section: 'learn',
		part: 1,
		chapter: 3,
		status: 'in-progress',
		difficulty: 'intro',
	},
	{
		slug: 'learn/autoregressive-generation',
		title: 'Autoregressive generation',
		description: 'Why inference is a loop, and a generate step you can read line by line.',
		section: 'learn',
		part: 1,
		chapter: 4,
		status: 'in-progress',
		difficulty: 'intro',
	},
	{
		slug: 'learn/prefill-decode-kv-cache',
		title: 'Prefill, decode, and the KV cache',
		description: 'Two phases of one request, what the cache stores, and why decode is bandwidth-heavy.',
		section: 'learn',
		part: 1,
		chapter: 5,
		status: 'in-progress',
		difficulty: 'intermediate',
		tags: ['prefill', 'decode', 'kv-cache', 'ttft'],
	},
	{
		slug: 'learn/attention-variants',
		title: 'MHA, MQA, GQA, and MLA',
		description: 'Attention variants and what they change for KV memory.',
		section: 'learn',
		part: 1,
		chapter: 6,
		status: 'in-progress',
		difficulty: 'intermediate',
	},
	{
		slug: 'learn/memory-footprint',
		title: 'Parameters, activations, and KV memory',
		description: 'What has to fit in HBM before a request can run.',
		section: 'learn',
		part: 2,
		chapter: 7,
		status: 'planned',
		difficulty: 'intro',
	},
	{
		slug: 'learn/inference-mathematics',
		title: 'Inference mathematics',
		description: 'FLOPs, weight memory, KV cache, arithmetic intensity, and a 7B-class worked example.',
		section: 'learn',
		part: 2,
		chapter: 8,
		status: 'planned',
		difficulty: 'intermediate',
		tags: ['flops', 'roofline', 'kv'],
	},
	{
		slug: 'learn/roofline',
		title: 'Roofline model',
		description: 'Compute-bound versus memory-bound regimes.',
		section: 'learn',
		part: 2,
		chapter: 9,
		status: 'planned',
		difficulty: 'intermediate',
	},
	{
		slug: 'learn/gpu-fundamentals',
		title: 'GPU fundamentals',
		description: 'SMs, warps, registers, shared memory, HBM, and tensor cores.',
		section: 'learn',
		part: 2,
		chapter: 10,
		status: 'planned',
		difficulty: 'intro',
	},
	{
		slug: 'learn/latency-throughput-goodput',
		title: 'Latency, throughput, and goodput',
		description: 'TTFT, ITL/TPOT, concurrency, and SLO-aware throughput.',
		section: 'learn',
		part: 2,
		chapter: 11,
		status: 'planned',
		difficulty: 'intro',
	},
	{
		slug: 'learn/benchmarking',
		title: 'Benchmarking methodology',
		description: 'How to measure inference without fooling yourself.',
		section: 'learn',
		part: 2,
		chapter: 12,
		status: 'planned',
		difficulty: 'intermediate',
	},
	{
		slug: 'learn/kv-cache',
		title: 'KV cache',
		description: 'Why keys and values can be stored, and what that costs in memory.',
		section: 'learn',
		part: 3,
		chapter: 13,
		status: 'planned',
		difficulty: 'intermediate',
		tags: ['kv-cache'],
	},
	{
		slug: 'learn/flashattention',
		title: 'FlashAttention',
		description: 'IO-aware attention kernels.',
		section: 'learn',
		part: 3,
		chapter: 14,
		status: 'planned',
		difficulty: 'advanced',
	},
	{
		slug: 'learn/paged-attention',
		title: 'PagedAttention',
		description: 'KV memory management without wasting a giant contiguous block per sequence.',
		section: 'learn',
		part: 3,
		chapter: 15,
		status: 'planned',
		difficulty: 'advanced',
	},
	{
		slug: 'learn/continuous-batching',
		title: 'Continuous batching and chunked prefill',
		description: 'Keep the GPU full without stranding short requests behind long prompts.',
		section: 'learn',
		part: 3,
		chapter: 16,
		status: 'planned',
		difficulty: 'intermediate',
	},
	{
		slug: 'learn/quantization',
		title: 'Quantization',
		description: 'FP16/BF16, FP8, INT8, INT4, and hardware constraints.',
		section: 'learn',
		part: 3,
		chapter: 17,
		status: 'planned',
		difficulty: 'intermediate',
	},
	{
		slug: 'learn/speculative-decoding',
		title: 'Speculative decoding',
		description: 'Draft tokens and verify them on the target model.',
		section: 'learn',
		part: 3,
		chapter: 18,
		status: 'planned',
		difficulty: 'advanced',
	},
	{
		slug: 'learn/kernel-fusion-cuda-graphs',
		title: 'Kernel fusion, compilation, and CUDA graphs',
		description: 'Cut launch overhead when decode is a thin loop.',
		section: 'learn',
		part: 3,
		chapter: 19,
		status: 'planned',
		difficulty: 'advanced',
	},
	{
		slug: 'learn/data-parallelism',
		title: 'Data parallelism and replica serving',
		description: 'Copy the model, split the traffic.',
		section: 'learn',
		part: 4,
		chapter: 20,
		status: 'planned',
		difficulty: 'intermediate',
	},
	{
		slug: 'learn/tensor-parallelism',
		title: 'Tensor parallelism',
		description: 'Split layers across GPUs when one card cannot hold the weights.',
		section: 'learn',
		part: 4,
		chapter: 21,
		status: 'planned',
		difficulty: 'advanced',
	},
	{
		slug: 'learn/pipeline-parallelism',
		title: 'Pipeline parallelism',
		description: 'Split depth across GPUs.',
		section: 'learn',
		part: 4,
		chapter: 22,
		status: 'planned',
		difficulty: 'advanced',
	},
	{
		slug: 'learn/expert-parallelism',
		title: 'Expert parallelism for MoE',
		description: 'Route tokens to experts that live on different devices.',
		section: 'learn',
		part: 4,
		chapter: 23,
		status: 'planned',
		difficulty: 'advanced',
	},
	{
		slug: 'learn/context-parallelism',
		title: 'Context and sequence parallelism',
		description: 'Split long sequences when attention state does not fit.',
		section: 'learn',
		part: 4,
		chapter: 24,
		status: 'planned',
		difficulty: 'advanced',
	},
	{
		slug: 'learn/nccl-collectives',
		title: 'NCCL and communication cost',
		description: 'What parallel strategies spend on the wire.',
		section: 'learn',
		part: 4,
		chapter: 25,
		status: 'planned',
		difficulty: 'advanced',
	},
	{
		slug: 'learn/pytorch-baseline',
		title: 'Minimal PyTorch inference baseline',
		description: 'A native generate loop before engines enter the picture.',
		section: 'learn',
		part: 5,
		chapter: 26,
		status: 'planned',
		difficulty: 'intermediate',
	},
	{
		slug: 'learn/caching-batching-scheduling',
		title: 'Caching, batching, and scheduling',
		description: 'Add the pieces that turn a loop into an engine.',
		section: 'learn',
		part: 5,
		chapter: 27,
		status: 'planned',
		difficulty: 'advanced',
	},
	{
		slug: 'learn/vllm-internals',
		title: 'vLLM internals',
		description: 'How a production engine compares to the baseline.',
		section: 'learn',
		part: 5,
		chapter: 28,
		status: 'planned',
		difficulty: 'advanced',
	},
	{
		slug: 'learn/sglang-trtllm',
		title: 'SGLang and TensorRT-LLM',
		description: 'Architecture overviews and trade-offs.',
		section: 'learn',
		part: 5,
		chapter: 29,
		status: 'planned',
		difficulty: 'advanced',
	},
	{
		slug: 'learn/engine-benchmarks',
		title: 'Engine benchmark methodology',
		description: 'Compare engines without mismatched setups.',
		section: 'learn',
		part: 5,
		chapter: 30,
		status: 'planned',
		difficulty: 'intermediate',
	},
	{
		slug: 'learn/gateway-registry-routing',
		title: 'Gateway, registry, and routing',
		description: 'API surface, health checks, and model lookup.',
		section: 'learn',
		part: 6,
		chapter: 31,
		status: 'planned',
		difficulty: 'intermediate',
	},
	{
		slug: 'learn/queues-admission-lb',
		title: 'Queues, admission control, and load balancing',
		description: 'Protect the GPU from overload and route around failure.',
		section: 'learn',
		part: 6,
		chapter: 32,
		status: 'planned',
		difficulty: 'intermediate',
	},
	{
		slug: 'learn/observability-slos',
		title: 'Metrics, tracing, logging, and SLOs',
		description: 'What to watch once traffic is real.',
		section: 'learn',
		part: 6,
		chapter: 33,
		status: 'planned',
		difficulty: 'intermediate',
	},
	{
		slug: 'learn/containers-k8s',
		title: 'Containers and Kubernetes',
		description: 'Deploy when the operational model justifies it.',
		section: 'learn',
		part: 6,
		chapter: 34,
		status: 'planned',
		difficulty: 'intermediate',
	},
	{
		slug: 'learn/autoscaling-fault-tolerance',
		title: 'Replicas, autoscaling, and fault tolerance',
		description: 'Stay up when a replica or a node disappears.',
		section: 'learn',
		part: 6,
		chapter: 35,
		status: 'planned',
		difficulty: 'advanced',
	},
	{
		slug: 'learn/capacity-and-cost',
		title: 'Capacity planning and cost per token',
		description: 'Utilization, goodput, and the bill.',
		section: 'learn',
		part: 6,
		chapter: 36,
		status: 'planned',
		difficulty: 'intermediate',
	},
	{
		slug: 'reference',
		title: 'Reference',
		description: 'Glossary, formulas, and tables. Calculators come later.',
		section: 'reference',
		status: 'draft',
	},
	{
		slug: 'journal',
		title: 'Build journal',
		description: 'Dated experiments, decisions, failures, and lessons.',
		section: 'journal',
		status: 'draft',
	},
];

export const LEARNING_TRACKS = [
	{
		id: 'fundamentals',
		title: 'Fundamentals',
		description: 'How a decoder generates tokens, and what the GPU is doing.',
		slugs: [
			'learn/from-text-to-tokens',
			'learn/transformer-forward-pass',
			'learn/prefill-decode-kv-cache',
			'learn/attention-variants',
		],
	},
	{
		id: 'optimization',
		title: 'Optimization',
		description: 'Make decode cheaper without changing the model card.',
		slugs: [
			'learn/kv-cache',
			'learn/continuous-batching',
			'learn/quantization',
			'learn/speculative-decoding',
		],
	},
	{
		id: 'production',
		title: 'Production serving',
		description: 'From a generate loop to a system you can operate.',
		slugs: [
			'learn/pytorch-baseline',
			'learn/vllm-internals',
			'learn/gateway-registry-routing',
			'learn/observability-slos',
		],
	},
];

export function inferenceV2Path(slug: string): string {
	if (!slug) return INFERENCE_V2_BASE;
	return `${INFERENCE_V2_BASE}/${slug}`;
}

/** `learn/01-transformer-inference` → `learn/transformer-inference` */
export function collectionIdToSlug(id: string): string {
	return id.replace(/\/index$/, '').replace(/\/(\d+)-/g, '/');
}

export function getNavPage(slug: string): NavPage | undefined {
	return NAV_PAGES.find((page) => page.slug === slug);
}

export function learnChapters(): NavPage[] {
	return NAV_PAGES.filter((page) => page.section === 'learn' && page.part != null);
}

export function chaptersByPart(): { part: number; title: string; chapters: NavPage[] }[] {
	return [1, 2, 3, 4, 5, 6].map((part) => ({
		part,
		title: PART_TITLES[part],
		chapters: learnChapters().filter((page) => page.part === part),
	}));
}

export function neighbors(slug: string): { prev: NavPage | null; next: NavPage | null } {
	const chapters = learnChapters();
	const index = chapters.findIndex((page) => page.slug === slug);
	if (index < 0) return { prev: null, next: null };
	return {
		prev: chapters[index - 1] ?? null,
		next: chapters[index + 1] ?? null,
	};
}

export type SearchDoc = {
	title: string;
	description: string;
	href: string;
	text: string;
	status: ChapterStatus;
};

export function buildSearchDocs(entries: CollectionEntry<'inferenceV2'>[]): SearchDoc[] {
	const bodies = new Map(entries.map((entry) => [collectionIdToSlug(entry.id), entry.body]));
	return NAV_PAGES.map((page) => ({
		title: page.title,
		description: page.description,
		href: inferenceV2Path(page.slug),
		text: stripForSearch(
			`${page.title} ${page.description} ${bodies.get(page.slug) ?? ''}`,
		),
		status: page.status,
	}));
}

export function stripForSearch(markdown: string): string {
	return markdown
		.replace(/```[\s\S]*?```/g, ' ')
		.replace(/\$\$[\s\S]*?\$\$/g, ' ')
		.replace(/\$[^$]+\$/g, ' ')
		.replace(/[#>*_`\[\]()]/g, ' ')
		.replace(/\s+/g, ' ')
		.trim();
}
