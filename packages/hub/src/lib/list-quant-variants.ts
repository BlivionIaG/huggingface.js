import {
	attentionConfigFromModelConfig,
	describeBaseVariant,
	getGGUFQuantBits,
	getGGUFVariantRows,
	kvCacheBytesPerToken,
	parseQuantVariantDescriptor,
	sumWeightFileSizes,
	type AttentionConfig,
	type QuantVariantRow,
} from "@huggingface/tasks";
import { HUB_URL } from "../consts";
import { createApiError } from "../error";
import type { ApiModelInfo } from "../types/api/api-model";
import type { CredentialsParams } from "../types/public";
import { checkCredentials } from "../utils/checkCredentials";
import { parseLinkHeader } from "../utils/parseLinkHeader";

/**
 * Parameters of {@link listQuantVariants}.
 */
export interface ListQuantVariantsParams {
	/**
	 * Id of the base model, e.g. `Qwen/Qwen3.8-27B`.
	 */
	name: string;
	/**
	 * Authors whose variants are listed.
	 *
	 * Defaults to the author of the base model, which is what the model page shows: the
	 * quantizations published by the organization that owns the model.
	 * Pass an empty array to list the variants of every author.
	 */
	authors?: string[] | "same";
	/**
	 * Maximum number of variant repos to inspect.
	 */
	limit?: number;
	/**
	 * Custom hub url, for a self hosted hub.
	 */
	hubUrl?: string;
	/**
	 * Custom fetch function to use instead of the default one, for example to use a proxy or edit headers.
	 */
	fetch?: typeof fetch;
}

interface FetchParams {
	hubUrl?: string;
	fetch?: typeof fetch;
	accessToken?: string;
}

async function fetchModelInfo(name: string, params: FetchParams): Promise<ApiModelInfo> {
	const search = new URLSearchParams([
		["blobs", "true"],
		["expand[]", "config"],
		["expand[]", "safetensors"],
		["expand[]", "cardData"],
		["expand[]", "gguf"],
	]).toString();

	const response = await (params.fetch || fetch)(`${params.hubUrl || HUB_URL}/api/models/${name}?${search}`, {
		headers: {
			accept: "application/json",
			...(params.accessToken ? { Authorization: `Bearer ${params.accessToken}` } : undefined),
		},
	});

	if (!response.ok) {
		throw await createApiError(response);
	}

	return (await response.json()) as ApiModelInfo;
}

/**
 * Reads the `config.json` of a repo.
 *
 * The `config` expansion of the model info holds a normalized version of the config, without
 * the dimensions of the model, so the raw file is needed to know the size of the KV cache.
 */
async function fetchModelConfig(name: string, params: FetchParams): Promise<unknown> {
	const response = await (params.fetch || fetch)(`${params.hubUrl || HUB_URL}/${name}/raw/main/config.json`, {
		headers: {
			accept: "application/json",
			...(params.accessToken ? { Authorization: `Bearer ${params.accessToken}` } : undefined),
		},
	});

	if (!response.ok) {
		return undefined;
	}

	try {
		return await response.json();
	} catch {
		return undefined;
	}
}

/**
 * Lists the repos that are quantized variants of a base model.
 *
 * Variants are the repos that declare the base model in their `base_model` metadata, which is
 * how the Hub links a quantization to the model it comes from (`Qwen/Qwen3.8-27B-FP8` declares
 * `base_model: ["Qwen/Qwen3.8-27B"]`).
 */
async function findVariantRepos(
	baseName: string,
	authors: string[],
	params: FetchParams & { limit: number },
): Promise<string[]> {
	const search = new URLSearchParams([
		["filter", `base_model:${baseName}`],
		["limit", String(Math.min(params.limit, 100))],
		["expand[]", "safetensors"],
		["expand[]", "cardData"],
	]);

	if (authors.length > 0) {
		search.set("author", authors.join(","));
	}

	let url: string | undefined = `${params.hubUrl || HUB_URL}/api/models?${search.toString()}`;
	const repos: string[] = [];

	while (url && repos.length < params.limit) {
		const response: Response = await (params.fetch || fetch)(url, {
			headers: {
				accept: "application/json",
				...(params.accessToken ? { Authorization: `Bearer ${params.accessToken}` } : undefined),
			},
		});

		if (!response.ok) {
			throw await createApiError(response);
		}

		const items = (await response.json()) as ApiModelInfo[];
		for (const item of items) {
			if (item.id !== baseName && repos.length < params.limit) {
				repos.push(item.id);
			}
		}

		const linkHeader = response.headers.get("Link");
		url = linkHeader ? parseLinkHeader(linkHeader).next : undefined;
	}

	return repos;
}

/**
 * Dtype of the biggest part of the weights of a model.
 */
function getDominantDtype(info: ApiModelInfo): string | undefined {
	const parameters = info.safetensors?.parameters;
	if (!parameters) {
		return undefined;
	}

	let dominant: string | undefined;
	let dominantCount = -1;
	for (const [dtype, count] of Object.entries(parameters)) {
		if (count > dominantCount) {
			dominant = dtype;
			dominantCount = count;
		}
	}
	return dominant;
}

/**
 * Builds the rows of a repo: one row per `.gguf` file for a GGUF repo, one row for the whole
 * repo otherwise.
 *
 * @param contextLength maximum context length of the model, shared by every quantization
 * @param kvCacheBytesPerToken size of the KV cache per token, shared by every quantization
 */
function getRowsForRepo(info: ApiModelInfo, contextLength?: number, kvCacheBytesPerToken?: number): QuantVariantRow[] {
	const files = info.siblings ?? [];

	const ggufRows = getGGUFVariantRows(info.id, files);
	if (ggufRows.length > 0) {
		return ggufRows.map((row) => ({
			...row,
			bits: getGGUFQuantBits(row.label),
			totalParams: info.safetensors?.total ?? info.gguf?.total,
			contextLength: info.gguf?.context_length ?? contextLength,
			kvCacheBytesPerToken,
			downloads: info.downloads,
			likes: info.likes,
		}));
	}

	const descriptor = parseQuantVariantDescriptor(info.id) ?? describeBaseVariant(getDominantDtype(info));
	if (!descriptor) {
		return [];
	}

	const { totalBytes } = sumWeightFileSizes(files);

	return [
		{
			id: info.id,
			label: descriptor.label,
			format: descriptor.format,
			bits: descriptor.bits,
			sizeBytes: totalBytes,
			totalParams: info.safetensors?.total,
			contextLength,
			kvCacheBytesPerToken,
			downloads: info.downloads,
			likes: info.likes,
		},
	];
}

/**
 * Lists the quantized variants of a model, as the rows of the quantization menu of a model page.
 *
 * The rows cover the base model itself and every quantization published for it: the GGUF
 * quantizations of a GGUF repo (one row per quantization) and the quantized repos published
 * for the model (`Qwen/Qwen3.8-27B-FP8` and friends).
 *
 * @example
 * ```ts
 * const rows = await listQuantVariants({ name: "Qwen/Qwen3.8-27B" });
 * // => [
 * //   { id: "Qwen/Qwen3.8-27B-FP8", label: "FP8", bits: 8, sizeBytes: 30879676248, ... },
 * //   { id: "Qwen/Qwen3.8-27B", label: "BF16", bits: 16, sizeBytes: 55646414848, ... },
 * // ]
 * ```
 */
export async function listQuantVariants(
	params: ListQuantVariantsParams & Partial<CredentialsParams>,
): Promise<QuantVariantRow[]> {
	const accessToken = params && checkCredentials(params);
	const fetchParams: FetchParams = { hubUrl: params.hubUrl, fetch: params.fetch, accessToken };

	const baseName = params.name;
	const baseAuthor = baseName.includes("/") ? baseName.split("/")[0] : undefined;
	const authors =
		params.authors === undefined || params.authors === "same" ? (baseAuthor ? [baseAuthor] : []) : params.authors;

	const limit = params.limit ?? 50;

	const baseInfo = await fetchModelInfo(baseName, fetchParams);

	// Every quantization of a model shares the architecture of the model, and so its KV cache.
	const modelConfig = await fetchModelConfig(baseName, fetchParams);
	const attentionConfig: AttentionConfig | undefined = attentionConfigFromModelConfig(modelConfig);
	const contextLength = baseInfo.gguf?.context_length ?? attentionConfig?.maxPositionEmbeddings;
	const kvCacheBytes = attentionConfig ? kvCacheBytesPerToken(attentionConfig) : undefined;

	const rows: QuantVariantRow[] = getRowsForRepo(baseInfo, contextLength, kvCacheBytes);

	const variantRepos = await findVariantRepos(baseName, authors, { ...fetchParams, limit });

	for (const id of variantRepos) {
		const details = await fetchModelInfo(id, fetchParams);
		rows.push(...getRowsForRepo(details, contextLength, kvCacheBytes));
	}

	// The menu is ordered like the GGUF menu of a model page: the smallest quantizations
	// first, grouped by the number of bits per weight.
	return rows.sort((a, b) => (a.bits ?? 16) - (b.bits ?? 16) || a.sizeBytes - b.sizeBytes);
}
