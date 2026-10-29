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
		slug: "learn/inference-problem",
		title: "The inference problem",
		description: "Training versus inference, the generation loop, latency/throughput/cost, memory bandwidth, and roofline.",
		part: 1,
		chapter: 0,
		section: 'learn',
		status: 'draft',
		difficulty: 'intermediate',
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
		title: "Parameters, activations, and KV memory",
		description: "Size weights, cache state, workspace, and concurrency using complete worked memory budgets.",
		section: 'learn',
		part: 2,
		chapter: 7,
		status: 'draft',
		difficulty: 'intro',
	},
	{
		slug: 'learn/inference-mathematics',
		title: "Inference mathematics",
		description: "Derive matrix FLOPs, arithmetic intensity, roofline, communication, latency, and capacity with worked examples.",
		section: 'learn',
		part: 2,
		chapter: 8,
		status: 'draft',
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
		status: 'draft',
		difficulty: 'intro',
	},
	{
		slug: 'learn/latency-throughput-goodput',
		title: "Latency, throughput, and goodput",
		description: "Streaming metrics, percentile latency, attention arithmetic intensity, and useful throughput under an SLO.",
		section: 'learn',
		part: 2,
		chapter: 11,
		status: 'draft',
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
		status: 'draft',
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
		status: 'draft',
		difficulty: 'advanced',
	},
	{
		slug: 'learn/paged-attention',
		title: 'PagedAttention',
		description: 'KV memory management without wasting a giant contiguous block per sequence.',
		section: 'learn',
		part: 3,
		chapter: 15,
		status: 'draft',
		difficulty: 'advanced',
	},
	{
		slug: 'learn/continuous-batching',
		title: 'Continuous batching and chunked prefill',
		description: 'Keep the GPU full without stranding short requests behind long prompts.',
		section: 'learn',
		part: 3,
		chapter: 16,
		status: 'draft',
		difficulty: 'intermediate',
	},
	{
		slug: 'learn/quantization',
		title: 'Quantization',
		description: 'FP16/BF16, FP8, INT8, INT4, and hardware constraints.',
		section: 'learn',
		part: 3,
		chapter: 17,
		status: 'draft',
		difficulty: 'intermediate',
	},
	{
		slug: 'learn/speculative-decoding',
		title: 'Speculative decoding',
		description: 'Draft tokens and verify them on the target model.',
		section: 'learn',
		part: 3,
		chapter: 18,
		status: 'draft',
		difficulty: 'advanced',
	},
	{
		slug: 'learn/kernel-fusion-cuda-graphs',
		title: 'Kernel fusion, compilation, and CUDA graphs',
		description: 'Cut launch overhead when decode is a thin loop.',
		section: 'learn',
		part: 3,
		chapter: 19,
		status: 'draft',
		difficulty: 'advanced',
	},
	{
		slug: 'learn/data-parallelism',
		title: "Data parallelism and replica serving",
		description: "Replicate a model across workers, route independent requests, and account for capacity and queue imbalance.",
		section: 'learn',
		part: 4,
		chapter: 20,
		status: 'draft',
		difficulty: 'intermediate',
	},
	{
		slug: 'learn/tensor-parallelism',
		title: 'Tensor parallelism',
		description: 'Split layers across GPUs when one card cannot hold the weights.',
		section: 'learn',
		part: 4,
		chapter: 21,
		status: 'draft',
		difficulty: 'advanced',
	},
	{
		slug: 'learn/pipeline-parallelism',
		title: 'Pipeline parallelism',
		description: 'Split depth across GPUs.',
		section: 'learn',
		part: 4,
		chapter: 22,
		status: 'draft',
		difficulty: 'advanced',
	},
	{
		slug: 'learn/expert-parallelism',
		title: 'Expert parallelism for MoE',
		description: 'Route tokens to experts that live on different devices.',
		section: 'learn',
		part: 4,
		chapter: 23,
		status: 'draft',
		difficulty: 'advanced',
	},
	{
		slug: 'learn/context-parallelism',
		title: 'Context and sequence parallelism',
		description: 'Split long sequences when attention state does not fit.',
		section: 'learn',
		part: 4,
		chapter: 24,
		status: 'draft',
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
		title: "A measured PyTorch inference baseline",
		description: "Compare cached and uncached generation with matched prompts, warmup, output checks, and explicit timing boundaries.",
		section: 'learn',
		part: 5,
		chapter: 26,
		status: 'draft',
		difficulty: 'intermediate',
	},
	{
		slug: 'learn/caching-batching-scheduling',
		title: "Inside a single-model serving engine",
		description: "Follow HTTP requests through sequence IDs, queues, batching, workers, streaming, and cluster replicas.",
		section: 'learn',
		part: 5,
		chapter: 27,
		status: 'draft',
		difficulty: 'advanced',
	},
	{
		slug: 'learn/vllm-internals',
		title: "vLLM from a generate call to streaming",
		description: "Load once, batch prompts, control sampling, and stream through an OpenAI-compatible server.",
		section: 'learn',
		part: 5,
		chapter: 28,
		status: 'draft',
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
		title: "Serving multiple models",
		description: "A model registry, LRU worker cache, API examples, cold starts, eviction, and cache-aware cluster routing.",
		section: 'learn',
		part: 6,
		chapter: 31,
		status: 'draft',
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
		status: 'draft',
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
		status: 'draft',
		difficulty: 'intermediate',
	},
	{
		slug: "learn/edge-inference",
		title: "Edge and on-device inference",
		description: "Memory budgeting, runtimes, thermal limits, and architecture choices.",
		part: 6,
		chapter: 37,
		section: 'learn',
		status: 'draft',
		difficulty: 'intermediate',
	},
	{
		slug: "learn/inference-time-scaling",
		title: "Inference-time compute scaling",
		description: "Reasoning budgets, verifiers, process rewards, and bounded search.",
		part: 6,
		chapter: 38,
		section: 'learn',
		status: 'draft',
		difficulty: 'intermediate',
	},
	{
		slug: "learn/future-of-inference",
		title: "The future of inference",
		description: "Diffusion, multi-token prediction, alternative hardware, and inference-first design.",
		part: 6,
		chapter: 39,
		section: 'learn',
		status: 'draft',
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

/** One sequence for the chapter list, sidebar, and previous/next links. */
export const BOOK_CHAPTERS = [
  {
    "slug": "learn/inference-problem",
    "title": "The inference problem",
    "description": "A user sends a prompt and waits for an answer."
  },
  {
    "slug": "learn/gpu-fundamentals",
    "title": "Hardware foundations for inference",
    "description": "The roofline model tells us which resource can limit a workload."
  },
  {
    "slug": "learn/transformer-forward-pass",
    "title": "Transformer inference mechanics",
    "description": "A request turns text into vectors, transforms those vectors through attention and feed-forward layers, and selects a token from vocabulary scores."
  },
  {
    "slug": "learn/quantization",
    "title": "Quantization",
    "description": "The model\u2019s weights are fixed during inference, but their representation can change."
  },
  {
    "slug": "learn/speculative-decoding",
    "title": "Speculative decoding",
    "description": "Small-batch decoding often leaves arithmetic capacity unused while the model streams weights."
  },
  {
    "slug": "learn/kv-cache",
    "title": "KV cache optimization",
    "description": "The cache saves computation, but it also becomes a growing memory commitment for every active request."
  },
  {
    "slug": "learn/flashattention",
    "title": "Kernel engineering and FlashAttention",
    "description": "Cache management determines which state remains available."
  },
  {
    "slug": "learn/continuous-batching",
    "title": "Serving systems architecture",
    "description": "A fast forward pass is one component of a service."
  },
  {
    "slug": "learn/data-parallelism",
    "title": "Parallelism for large model inference",
    "description": "A model can exceed one device\u2019s capacity, or a service can outgrow one worker\u2019s throughput."
  },
  {
    "slug": "learn/context-parallelism",
    "title": "Long-context and memory management",
    "description": "Long context increases both retained state and attention work."
  },
  {
    "slug": "learn/edge-inference",
    "title": "Edge and on-device inference",
    "description": "On-device inference brings computation closer to the user and can support offline use."
  },
  {
    "slug": "learn/inference-time-scaling",
    "title": "Inference-time compute scaling",
    "description": "Efficient execution makes each unit of model work cheaper."
  },
  {
    "slug": "learn/observability-slos",
    "title": "Observability and production engineering",
    "description": "Production adds responsibilities beyond generating correct tokens: requests must complete reliably, latency must remain within targets, and resource costs must be attributable."
  },
  {
    "slug": "learn/future-of-inference",
    "title": "The future of inference",
    "description": "New generation methods and hardware should be evaluated against the same question: how much useful output do they deliver within a latency, memory, energy, and quality budget? This chapter considers changes to the generation algorithm and to the architecture itself.."
  }
];

export const CHAPTER_REDIRECTS: Record<string, string> = {
  "learn/memory-footprint": "learn/gpu-fundamentals",
  "learn/inference-mathematics": "learn/gpu-fundamentals",
  "learn/from-text-to-tokens": "learn/transformer-forward-pass",
  "learn/logits-and-sampling": "learn/transformer-forward-pass",
  "learn/autoregressive-generation": "learn/transformer-forward-pass",
  "learn/prefill-decode-kv-cache": "learn/transformer-forward-pass",
  "learn/attention-variants": "learn/transformer-forward-pass",
  "learn/pytorch-baseline": "learn/transformer-forward-pass",
  "learn/paged-attention": "learn/kv-cache",
  "learn/kernel-fusion-cuda-graphs": "learn/flashattention",
  "learn/caching-batching-scheduling": "learn/continuous-batching",
  "learn/vllm-internals": "learn/continuous-batching",
  "learn/tensor-parallelism": "learn/data-parallelism",
  "learn/pipeline-parallelism": "learn/data-parallelism",
  "learn/expert-parallelism": "learn/data-parallelism",
  "learn/gateway-registry-routing": "learn/observability-slos",
  "learn/latency-throughput-goodput": "learn/observability-slos",
  "learn/capacity-and-cost": "learn/observability-slos"
};

export function learnChapters(): NavPage[] {
 return BOOK_CHAPTERS.map((chapter, index) => ({ ...chapter, section: "learn", part: 1, status: "complete", chapter: index + 1 }));
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
