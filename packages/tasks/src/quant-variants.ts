import { parseGGUFQuantLabel } from "./gguf.js";

/**
 * Quantization formats a model (or a model variant) can be published in.
 *
 * `gguf` is a special case: the quantization of a GGUF artifact is carried by the *file* name
 * (e.g. `Q4_K_M`), not by the repo name, so a GGUF repo exposes one variant per `.gguf` file.
 * Every other format is a property of the whole repo (`Qwen/Qwen3.8-27B-FP8` is an FP8 repo).
 */
export type QuantVariantFormat =
	| "gguf"
	| "fp32"
	| "fp16"
	| "bf16"
	| "fp8"
	| "int8"
	| "int6"
	| "int5"
	| "int4"
	| "int3"
	| "int2"
	| "awq"
	| "gptq"
	| "nvfp4"
	| "mxfp4"
	| "mxfp8"
	| "nf4"
	| "exl3"
	| "mlx"
	| "w4a16";

/**
 * How a quantization variant is identified, from the name of the repo that hosts it.
 */
export interface QuantVariantDescriptor {
	/**
	 * Display label, e.g. `FP8`, `AWQ INT4`, `MLX 4bit`.
	 */
	label: string;
	/**
	 * Quantization family.
	 */
	format: QuantVariantFormat;
	/**
	 * Nominal number of bits per weight, when the format has a well defined one.
	 */
	bits?: number;
}

/**
 * A single downloadable artifact of a model: either a whole repo (safetensors based variants)
 * or one file of a repo (GGUF variants).
 *
 * This is the same shape as the rows of the GGUF menu of a model page, which lets the two
 * menus be rendered by the same component.
 */
export interface QuantVariantRow {
	/**
	 * Repo the artifact belongs to.
	 */
	id: string;
	/**
	 * File of the repo, when the row points to a single file (GGUF artifacts).
	 */
	file?: string;
	/**
	 * Files of the repo that hold the artifact, when it is split in several files.
	 */
	files?: string[];
	/**
	 * Display label of the variant, e.g. `FP8`, `Q4_K_M`.
	 */
	label: string;
	/**
	 * Quantization family of the variant.
	 */
	format: QuantVariantFormat;
	/**
	 * Nominal number of bits per weight.
	 */
	bits?: number;
	/**
	 * Size of the weights, in bytes.
	 */
	sizeBytes: number;
	/**
	 * Role of the artifact inside the repo.
	 */
	role?: QuantVariantRole;
	/**
	 * Total number of parameters of the model, when known.
	 */
	totalParams?: number;
	/**
	 * Maximum context length of the model, when known.
	 */
	contextLength?: number;
	/**
	 * Size of the KV cache of the model, per token of context.
	 *
	 * Every quantization of a model shares the same KV cache: it depends on the architecture,
	 * not on the way the weights are stored.
	 */
	kvCacheBytesPerToken?: number;
	/**
	 * Number of downloads of the repo the artifact belongs to.
	 */
	downloads?: number;
	/**
	 * Number of likes of the repo the artifact belongs to.
	 */
	likes?: number;
}

/**
 * Role of an artifact inside a repo.
 *
 * - `model`: the model weights themselves.
 * - `mmproj`: a multimodal projector, needed next to the weights of a vision model.
 * - `mtp`: a multi-token-prediction head, an optional extra file.
 * - `imatrix`: the quantization matrix of a repo, which is not a model.
 */
export type QuantVariantRole = "model" | "mmproj" | "mtp" | "imatrix";

/**
 * Bits-per-weight tokens, ordered from the most to the least specific.
 *
 * The microscaling formats are tested first because they contain the plain FP tokens:
 * `NVFP4` contains `FP4` and `MXFP8` contains `FP8`.
 */
const BIT_TOKENS: Array<{ re: RegExp; label: string; format: QuantVariantFormat; bits: number }> = [
	{ re: /\bNVFP4\b/, label: "NVFP4", format: "nvfp4", bits: 4 },
	{ re: /\bMXFP4\b/, label: "MXFP4", format: "mxfp4", bits: 4 },
	{ re: /\bMXFP8\b/, label: "MXFP8", format: "mxfp8", bits: 8 },
	{ re: /\bW4A16\b/, label: "W4A16", format: "w4a16", bits: 4 },
	{ re: /\bFP8\b|\bF8_E[45]M[23]\b|\bW8A8\b/, label: "FP8", format: "fp8", bits: 8 },
	{ re: /\bBF16\b/, label: "BF16", format: "bf16", bits: 16 },
	{ re: /\bFP16\b|\bFLOAT16\b/, label: "FP16", format: "fp16", bits: 16 },
	{ re: /\bINT8\b/, label: "INT8", format: "int8", bits: 8 },
	{ re: /\b8[- ]?BIT\b/, label: "8bit", format: "int8", bits: 8 },
	{ re: /\bINT6\b/, label: "INT6", format: "int6", bits: 6 },
	{ re: /\b6[- ]?BIT\b/, label: "6bit", format: "int6", bits: 6 },
	{ re: /\bINT5\b/, label: "INT5", format: "int5", bits: 5 },
	{ re: /\b5[- ]?BIT\b/, label: "5bit", format: "int5", bits: 5 },
	{ re: /\bINT4\b/, label: "INT4", format: "int4", bits: 4 },
	{ re: /\b4[- ]?BIT\b/, label: "4bit", format: "int4", bits: 4 },
	{ re: /\bINT3\b/, label: "INT3", format: "int3", bits: 3 },
	{ re: /\b3[- ]?BIT\b/, label: "3bit", format: "int3", bits: 3 },
	{ re: /\bINT2\b/, label: "INT2", format: "int2", bits: 2 },
	{ re: /\b2[- ]?BIT\b/, label: "2bit", format: "int2", bits: 2 },
];

/**
 * Quantization schemes, which qualify the bit width when both are present
 * (`Qwen3.8-27B-AWQ-INT4` is an `AWQ INT4` variant).
 */
const SCHEME_TOKENS: Array<{ re: RegExp; label: string; format: QuantVariantFormat }> = [
	{ re: /\bGGUF\b/, label: "GGUF", format: "gguf" },
	{ re: /\bAWQ\b/, label: "AWQ", format: "awq" },
	{ re: /\bGPTQ\b/, label: "GPTQ", format: "gptq" },
	{ re: /\bEXL3\b/, label: "EXL3", format: "exl3" },
	{ re: /\bNF4\b|\bBNB\b|BITSANDBYTES/, label: "NF4", format: "nf4" },
	{ re: /\bMLX\b/, label: "MLX", format: "mlx" },
];

/**
 * Identifies the quantization variant published in a repo, from its name.
 *
 * The name is expected to hold the variant as a suffix, which is the convention followed by
 * the Hub (`Qwen/Qwen3.8-27B-FP8`, `mlx-community/Qwen3.8-27B-4bit`, …).
 *
 * The number of bits is read from the name of the repo only, while the quantization scheme is
 * read from the whole id: some publishers put the scheme in the author (`mlx-community`).
 *
 * @returns the descriptor of the variant, or `undefined` when the name doesn't hold one
 * (e.g. a plain fine-tune like `Qwen/Qwen3.8-27B-Instruct`).
 */
export function parseQuantVariantDescriptor(repoName: string): QuantVariantDescriptor | undefined {
	const name = repoName.split("/").pop() ?? repoName;
	const upperName = name.toUpperCase();
	const upperId = repoName.toUpperCase();

	const bit = BIT_TOKENS.find((token) => token.re.test(upperName));
	const scheme = SCHEME_TOKENS.find((token) => token.re.test(upperId));

	if (!bit && !scheme) {
		return undefined;
	}

	if (bit) {
		// The microscaling formats are complete on their own: `NVFP4` is not an `INT4` variant.
		if (["nvfp4", "mxfp4", "mxfp8"].includes(bit.format)) {
			return { label: bit.label, format: bit.format, bits: bit.bits };
		}

		if (scheme) {
			return { label: `${scheme.label} ${bit.label}`, format: scheme.format, bits: bit.bits };
		}

		return { label: bit.label, format: bit.format, bits: bit.bits };
	}

	if (scheme) {
		return { label: scheme.label, format: scheme.format };
	}

	return undefined;
}

/**
 * Describes the non-quantized weights of a model, from the dtype of its tensors.
 *
 * The BF16 row is what the top of the GGUF menu shows for a 16-bit release.
 */
export function describeBaseVariant(dtype?: string): QuantVariantDescriptor | undefined {
	switch (dtype?.toLowerCase()) {
		case "bfloat16":
		case "bf16":
			return { label: "BF16", format: "bf16", bits: 16 };
		case "float16":
		case "f16":
		case "half":
			return { label: "FP16", format: "fp16", bits: 16 };
		case "float32":
		case "f32":
			return { label: "FP32", format: "fp32", bits: 32 };
		case "float64":
		case "f64":
			return { label: "FP64", format: "fp32", bits: 64 };
		default:
			return undefined;
	}
}

/**
 * Extensions of the files that hold the weights of a model.
 *
 * Everything else in a repo (config, tokenizer, images, documentation) is not part of the
 * size of the model, and must not be counted when checking whether a model fits in a GPU.
 */
export const WEIGHT_FILE_EXTENSIONS = [
	".safetensors",
	".gguf",
	".bin",
	".pt",
	".pth",
	".ckpt",
	".exl3",
	".npz",
] as const;

/**
 * A file of a repo, as returned by the Hub API.
 */
export interface RepoFile {
	rfilename: string;
	size?: number;
}

/**
 * Whether a file holds the weights of a model.
 *
 * The index of a sharded safetensors repo (`.safetensors.index.json`) is not a weight file.
 */
export function isWeightFile(path: string): boolean {
	const lower = path.toLowerCase();
	return WEIGHT_FILE_EXTENSIONS.some((extension) => lower.endsWith(extension));
}

/**
 * Role of a GGUF file, deduced from its name.
 */
export function getQuantVariantRole(path: string): QuantVariantRole {
	const lower = path.toLowerCase();
	if (lower.includes("mmproj")) {
		return "mmproj";
	}
	if (lower.includes("mtp")) {
		return "mtp";
	}
	if (lower.includes("imatrix")) {
		return "imatrix";
	}
	return "model";
}

/**
 * Sums the size of the weight files of a repo.
 *
 * This is the exact size of the model as it is downloaded, which is the number a user compares
 * against the memory of their hardware.
 */
export function sumWeightFileSizes(files: RepoFile[]): {
	totalBytes: number;
	files: Array<{ path: string; sizeBytes: number }>;
} {
	const weightFiles = files
		.filter((file) => isWeightFile(file.rfilename))
		.map((file) => ({ path: file.rfilename, sizeBytes: file.size ?? 0 }));

	return {
		totalBytes: weightFiles.reduce((total, file) => total + file.sizeBytes, 0),
		files: weightFiles,
	};
}

/**
 * Key that identifies the shards of a same quantization.
 *
 * `Qwen3.8-27B-BF16-00001-of-00002.gguf` and `Qwen3.8-27B-BF16-00002-of-00002.gguf` are the two
 * shards of a same quantization, while two files that share a quantization but not a name are
 * two different models.
 */
function getGGUFShardKey(path: string): string {
	return path.replace(/-\d{5}-of-\d{5}(?=\.gguf$)/i, "");
}

/**
 * Builds the rows of the menu of a GGUF repo: one row per quantization.
 *
 * The label of a row is the quantization carried by the name of the file (`Q4_K_M`, `UD-IQ1_S`, …).
 * A quantization split in several files is a single row, whose size is the sum of its files.
 *
 * The quantization matrix of a repo (`imatrix_unsloth.gguf`) is not a model, and is left out.
 */
export function getGGUFVariantRows(id: string, files: RepoFile[]): QuantVariantRow[] {
	const groups = new Map<string, { role: QuantVariantRole; label: string; files: RepoFile[] }>();

	for (const file of files) {
		if (!file.rfilename.toLowerCase().endsWith(".gguf")) {
			continue;
		}

		const role = getQuantVariantRole(file.rfilename);
		if (role === "imatrix") {
			continue;
		}

		const quantLabel = parseGGUFQuantLabel(file.rfilename) ?? "GGUF";
		const label = role === "model" ? quantLabel : `${role === "mtp" ? "MTP" : "mmproj"} ${quantLabel}`;

		const key = `${role}:${label}:${getGGUFShardKey(file.rfilename)}`;
		const group = groups.get(key) ?? { role, label, files: [] };
		group.files.push(file);
		groups.set(key, group);
	}

	return [...groups.values()].map((group) => ({
		id,
		file: group.files.length === 1 ? group.files[0].rfilename : undefined,
		files: group.files.map((file) => file.rfilename),
		label: group.label,
		format: "gguf" as const,
		sizeBytes: group.files.reduce((total, file) => total + (file.size ?? 0), 0),
		role: group.role,
	}));
}

/**
 * The architecture of a model, as far as the KV cache is concerned.
 */
export interface AttentionConfig {
	/**
	 * Number of layers of the model.
	 */
	numHiddenLayers?: number;
	/**
	 * Number of attention heads.
	 */
	numAttentionHeads?: number;
	/**
	 * Number of key/value heads. Equal to the number of attention heads when the model does not
	 * use grouped-query attention.
	 */
	numKeyValueHeads?: number;
	/**
	 * Dimension of a head. Derived from `hiddenSize / numAttentionHeads` when not set.
	 */
	headDim?: number;
	/**
	 * Size of the hidden state.
	 */
	hiddenSize?: number;
	/**
	 * Maximum context length of the model.
	 */
	maxPositionEmbeddings?: number;
	/**
	 * Type of each layer, for hybrid models (`linear_attention` / `full_attention`).
	 */
	layerTypes?: string[];
}

/**
 * Reads the attention configuration of a model from its `config.json`.
 *
 * Both the flat form (`num_hidden_layers`) and the nested form used by multimodal models
 * (`text_config.num_hidden_layers`) are supported.
 */
export function attentionConfigFromModelConfig(config: unknown): AttentionConfig | undefined {
	if (typeof config !== "object" || config === null) {
		return undefined;
	}

	const root = config as Record<string, unknown>;
	const nested = (root.text_config ?? root.llm_config ?? root.language_config) as Record<string, unknown> | undefined;
	const source = { ...root, ...(nested ?? {}) };

	const num = (key: string): number | undefined => {
		const value = source[key];
		return typeof value === "number" ? value : undefined;
	};

	const numHiddenLayers = num("num_hidden_layers");
	const numAttentionHeads = num("num_attention_heads");
	if (numHiddenLayers === undefined && numAttentionHeads === undefined) {
		return undefined;
	}

	const layerTypes = Array.isArray(source.layer_types)
		? source.layer_types.filter((type): type is string => typeof type === "string")
		: undefined;

	return {
		numHiddenLayers,
		numAttentionHeads,
		numKeyValueHeads: num("num_key_value_heads"),
		headDim: num("head_dim"),
		hiddenSize: num("hidden_size"),
		maxPositionEmbeddings: num("max_position_embeddings"),
		layerTypes,
	};
}

/**
 * Number of bytes of KV cache held per token of context.
 *
 * Only the layers that use full attention hold a KV cache: the state of the linear attention
 * layers of hybrid models doesn't grow with the context (it is a few tens of MB for the whole
 * model, against the GBs of the KV cache).
 *
 * @param config attention configuration of the model
 * @param bytesPerElement size of a KV element, 2 bytes (FP16) by default
 */
export function kvCacheBytesPerToken(config: AttentionConfig, bytesPerElement = 2): number {
	const layers = config.layerTypes
		? config.layerTypes.filter((type) => type === "full_attention").length
		: (config.numHiddenLayers ?? 0);

	const kvHeads = config.numKeyValueHeads ?? config.numAttentionHeads ?? 0;
	const headDim =
		config.headDim ??
		(config.hiddenSize && config.numAttentionHeads ? config.hiddenSize / config.numAttentionHeads : 0);

	// K and V
	return 2 * layers * kvHeads * headDim * bytesPerElement;
}

/**
 * Breakdown of the memory needed to run a model.
 */
export interface MemoryRequirement {
	/**
	 * Size of the weights.
	 */
	weightBytes: number;
	/**
	 * Size of the KV cache, for the requested context length.
	 */
	kvCacheBytes: number;
	/**
	 * Runtime overhead (activations, CUDA context, …).
	 */
	overheadBytes: number;
	/**
	 * Total memory needed.
	 */
	totalBytes: number;
}

/**
 * Default ratio of the runtime overhead, over the size of the weights.
 */
export const DEFAULT_OVERHEAD_RATIO = 0.05;

/**
 * Default minimum runtime overhead, in bytes.
 */
export const DEFAULT_MIN_OVERHEAD_BYTES = 512 * 1024 * 1024;

/**
 * Estimates the memory needed to run a model.
 *
 * @param weightBytes size of the weights
 * @param kvCacheBytesPerToken size of the KV cache per token, 0 when the model has no KV cache
 * @param contextLength length of the context the model is run with
 */
export function estimateMemoryRequirement(params: {
	weightBytes: number;
	kvCacheBytesPerToken?: number;
	contextLength?: number;
	overheadRatio?: number;
	minOverheadBytes?: number;
}): MemoryRequirement {
	const { weightBytes } = params;
	const kvCacheBytes = (params.kvCacheBytesPerToken ?? 0) * (params.contextLength ?? 0);
	const overheadRatio = params.overheadRatio ?? DEFAULT_OVERHEAD_RATIO;
	const minOverheadBytes = params.minOverheadBytes ?? DEFAULT_MIN_OVERHEAD_BYTES;
	const overheadBytes = Math.max(minOverheadBytes, Math.round(weightBytes * overheadRatio));

	return {
		weightBytes,
		kvCacheBytes,
		overheadBytes,
		totalBytes: weightBytes + kvCacheBytes + overheadBytes,
	};
}

/**
 * How a model fits in the memory of a hardware.
 *
 * - `fits`: the model runs with the requested context, and leaves room to spare.
 * - `tight`: the model runs, but either it uses most of the memory, or it needs a shorter
 *   context than the one requested.
 * - `no-fit`: the weights of the model don't fit in the memory at all.
 */
export type FitState = "fits" | "tight" | "no-fit";

/**
 * Result of the comparison between the memory needed by a model and the memory of a hardware.
 */
export interface FitResult {
	state: FitState;
	/**
	 * Ratio of the memory needed over the memory available.
	 */
	ratio: number;
	/**
	 * Memory needed, in bytes.
	 */
	requiredBytes: number;
	/**
	 * Memory available, in bytes.
	 */
	availableBytes: number;
	/**
	 * Whether the weights of the model fit in the memory, whatever the context.
	 */
	weightsFit: boolean;
}

/**
 * Ratio under which a model is considered to run tightly.
 */
export const TIGHT_FIT_RATIO = 0.9;

/**
 * Nominal bits per weight of the GGUF quantization types.
 */
const GGUF_QUANT_BITS: Record<string, number> = {
	F32: 32,
	BF16: 16,
	F16: 16,
	Q8_K: 8,
	Q8_0: 8,
	Q8_1: 8,
	Q6_K: 6,
	Q5_K: 5,
	Q5_0: 5,
	Q5_1: 5,
	Q4_K: 4,
	IQ4_NL: 4,
	IQ4_XS: 4,
	Q4_0: 4,
	Q4_1: 4,
	MXFP4: 4,
	MXFP4_MOE: 4,
	NVFP4: 4,
	Q3_K: 3,
	IQ3_XXS: 3,
	IQ3_XS: 3,
	IQ3_S: 3,
	IQ3_M: 3,
	Q2_K: 2,
	IQ2_XXS: 2,
	IQ2_XS: 2,
	IQ2_S: 2,
	IQ2_M: 2,
	TQ2_0: 2,
	Q2_0: 2,
	IQ1_S: 1,
	IQ1_M: 1,
	TQ1_0: 1,
	Q1_0: 1,
};

/**
 * Nominal bits per weight of a GGUF quantization, from its label.
 *
 * The label of a GGUF artifact carries the quantization (`Q4_K_M`), sometimes prefixed by the
 * vendor (`UD-IQ1_S`) and suffixed by a size variation (`Q2_K_XL`).
 *
 * @returns the number of bits per weight, or `undefined` when the label is not a known quantization
 */
export function getGGUFQuantBits(label: string): number | undefined {
	const upper = label.toUpperCase();

	for (const candidate of [upper, upper.replace(/^UD-/, "")]) {
		const exact = GGUF_QUANT_BITS[candidate];
		if (exact !== undefined) {
			return exact;
		}
		// `Q2_K_XL` is a `Q2_K`, `IQ2_XXS` is an `IQ2_XXS`.
		const withoutVariation = candidate.replace(/_(XL|XXS|XS|S|M|L)$/, "");
		const variation = GGUF_QUANT_BITS[withoutVariation];
		if (variation !== undefined) {
			return variation;
		}
	}

	return undefined;
}

/**
 * Compares the memory needed by a model with the memory of a hardware.
 *
 * The weights and the runtime overhead have to fit for the model to run at all. The KV cache
 * grows with the context, so a model can fit in the memory of a hardware with a shorter context
 * than the one it was measured with.
 */
export function computeFit(requirement: MemoryRequirement, availableBytes: number): FitResult {
	const loadedBytes = requirement.weightBytes + requirement.overheadBytes;
	const ratio = availableBytes > 0 ? requirement.totalBytes / availableBytes : Infinity;
	const weightsFit = loadedBytes <= availableBytes;

	let state: FitState;
	if (!weightsFit) {
		state = "no-fit";
	} else if (requirement.totalBytes > availableBytes || ratio >= TIGHT_FIT_RATIO) {
		state = "tight";
	} else {
		state = "fits";
	}

	return {
		state,
		ratio,
		requiredBytes: requirement.totalBytes,
		availableBytes,
		weightsFit,
	};
}

/**
 * Estimates the memory needed to run a variant of a model.
 *
 * @param row variant of the menu, as returned by the Hub client
 * @param options context length to run the model with, defaults to the maximum context length
 * of the model. The KV cache grows with the context, so a shorter context needs less memory.
 */
export function getQuantVariantMemoryRequirement(
	row: QuantVariantRow,
	options?: { contextLength?: number; overheadRatio?: number; minOverheadBytes?: number },
): MemoryRequirement {
	return estimateMemoryRequirement({
		weightBytes: row.sizeBytes,
		kvCacheBytesPerToken: row.kvCacheBytesPerToken,
		contextLength: options?.contextLength ?? row.contextLength ?? 0,
		overheadRatio: options?.overheadRatio,
		minOverheadBytes: options?.minOverheadBytes,
	});
}

/**
 * Formats a size the way model pages do, e.g. `16.1 GB`.
 *
 * Sizes below 10 GB keep two decimals, bigger ones keep one, and whole sizes keep none.
 */
export function formatBytes(bytes: number): string {
	if (bytes < 1000) {
		return `${bytes} B`;
	}

	let value = bytes;
	let unit = "B";
	for (const next of ["kB", "MB", "GB", "TB"]) {
		value /= 1000;
		unit = next;
		if (value < 1000) {
			break;
		}
	}

	const decimals = value < 10 ? 2 : value < 100 ? 1 : 0;
	return `${Number.isInteger(value) ? value.toString() : value.toFixed(decimals)} ${unit}`;
}
