/**
 * Dumps the responses of the Hub API that the quantization menu needs, to the fixtures directory.
 *
 * The fixtures are served by `mock-hub.ts`, which lets the menu be tested offline, and without
 * hitting the Hub.
 *
 * The models to dump are given on the command line. The variants published for them are
 * discovered from the Hub, the way the menu discovers them, so that any model can be dumped
 * without listing its variants by hand.
 *
 * Usage:
 *   pnpm --filter @huggingface/hub dump-fixtures -- org/model
 *   pnpm --filter @huggingface/hub dump-fixtures -- org/model other/model
 *   pnpm --filter @huggingface/hub dump-fixtures -- org/model --limit 100
 */

import { mkdir, readdir, rm, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { parseArgs } from "node:util";
import { parseLinkHeader } from "../src/utils/parseLinkHeader.js";
import { stripArgumentSeparator } from "./args.js";

stripArgumentSeparator();

const HUB_URL = process.env.HF_HUB_URL ?? "https://huggingface.co";
const FIXTURES_DIR = join(__dirname, "fixtures");

/**
 * Number of variants captured for a model.
 *
 * It is the number of variants the menu reads by default, so that the menu of the mock hub is
 * the menu of the Hub.
 */
const DEFAULT_LIMIT = 50;

/**
 * Fields of a config the menu reads: the dimensions of the model, and its context length.
 */
const CONFIG_FIELDS = [
	"num_hidden_layers",
	"num_attention_heads",
	"num_key_value_heads",
	"head_dim",
	"hidden_size",
	"max_position_embeddings",
	"layer_types",
	"sliding_window",
	"use_sliding_window",
];

/**
 * Repos that are variants of a model.
 *
 * A variant declares the model it comes from in its `base_model` metadata, which is how the Hub
 * links a quantization to the model it comes from. The list is the one the menu reads.
 */
async function findVariantRepos(baseName: string, limit: number): Promise<string[]> {
	const search = new URLSearchParams([
		["filter", `base_model:${baseName}`],
		["limit", "100"],
		["expand[]", "safetensors"],
		["expand[]", "cardData"],
	]);

	let url: string | undefined = `${HUB_URL}/api/models?${search.toString()}`;
	const repos: string[] = [];

	while (url && repos.length < limit) {
		const response = await fetchWithRetry(url);
		if (!response.ok) {
			throw new Error(`Could not list the variants of ${baseName}: ${response.status} ${response.statusText}`);
		}

		const items = (await response.json()) as Array<{ id: string }>;
		for (const item of items) {
			if (item.id !== baseName && repos.length < limit) {
				repos.push(item.id);
			}
		}

		const linkHeader = response.headers.get("Link");
		url = linkHeader ? parseLinkHeader(linkHeader).next : undefined;
	}

	return repos;
}

/**
 * Fetches a url, waiting when the Hub is rate limiting the requests.
 */
async function fetchWithRetry(url: string): Promise<Response> {
	for (let attempt = 0; ; attempt++) {
		const response = await fetch(url);
		if ((response.status !== 429 && response.status < 500) || attempt >= 3) {
			return response;
		}

		const retryAfter = Number(response.headers.get("Retry-After"));
		const wait = Number.isFinite(retryAfter) && retryAfter > 0 ? retryAfter : 2 ** attempt;
		console.log(`  ${response.status} ${response.statusText}, waiting ${wait}s`);
		await new Promise((resolve) => setTimeout(resolve, wait * 1000));
	}
}

/**
 * Name of the file of a repo, `/` is not a valid character in a file name.
 */
function fixtureName(repo: string): string {
	return repo.replaceAll("/", "__");
}

/**
 * Keeps the fields of a model info the mock hub serves.
 *
 * The fixtures stay small and readable, so they can be edited to test a case.
 */
function pruneModelInfo(info: Record<string, unknown>): Record<string, unknown> {
	const siblings = ((info.siblings ?? []) as Array<Record<string, unknown>>).map((file) => ({
		rfilename: file.rfilename,
		size: file.size,
	}));

	const kept = [
		"_id",
		"id",
		"author",
		"cardData",
		"createdAt",
		"downloads",
		"downloadsAllTime",
		"gated",
		"gguf",
		"lastModified",
		"likes",
		"library_name",
		"pipeline_tag",
		"private",
		"safetensors",
		"tags",
		"trendingScore",
		"usedStorage",
	] as const;

	const pruned: Record<string, unknown> = { siblings };
	for (const key of kept) {
		if (info[key] !== undefined) {
			pruned[key] = info[key];
		}
	}
	return pruned;
}

/**
 * Keeps the fields of a config the menu reads.
 *
 * A config holds the vocabulary of the model, which is big and of no use to the menu.
 */
function pruneConfig(config: unknown): unknown {
	if (typeof config !== "object" || config === null) {
		return config;
	}

	const root = config as Record<string, unknown>;
	const nestedKey = ["text_config", "llm_config", "language_config"].find(
		(key) => typeof root[key] === "object" && root[key] !== null,
	);

	const kept: Record<string, unknown> = {};
	for (const key of CONFIG_FIELDS) {
		if (root[key] !== undefined) {
			kept[key] = root[key];
		}
	}

	if (nestedKey) {
		const nested = root[nestedKey] as Record<string, unknown>;
		const keptNested: Record<string, unknown> = {};
		for (const key of CONFIG_FIELDS) {
			if (nested[key] !== undefined) {
				keptNested[key] = nested[key];
			}
		}
		kept[nestedKey] = keptNested;
	}

	return kept;
}

/**
 * Dumps the info and the config of a repo.
 *
 * A repo the Hub does not answer for is left out, so that one broken variant does not cost all
 * the other ones.
 */
async function dumpRepo(repo: string): Promise<boolean> {
	const infoResponse = await fetchWithRetry(
		`${HUB_URL}/api/models/${repo}?blobs=true&expand[]=config&expand[]=safetensors&expand[]=cardData&expand[]=gguf`,
	);
	if (!infoResponse.ok) {
		console.log(`${repo}: skipped (${infoResponse.status} ${infoResponse.statusText})`);
		return false;
	}
	const info = await infoResponse.json();

	// The raw config holds the dimensions of the model, which the `config` expansion does not.
	const configResponse = await fetchWithRetry(`${HUB_URL}/${repo}/raw/main/config.json`);
	const config = configResponse.ok ? await configResponse.json() : undefined;

	await writeFile(
		join(FIXTURES_DIR, "models", `${fixtureName(repo)}.json`),
		JSON.stringify(pruneModelInfo(info), null, "\t"),
		"utf8",
	);
	await writeFile(
		join(FIXTURES_DIR, "configs", `${fixtureName(repo)}.json`),
		JSON.stringify({ id: repo, config: pruneConfig(config) }, null, "\t"),
		"utf8",
	);

	console.log(`${repo}: ${info.siblings?.length ?? 0} files`);
	return true;
}

async function main() {
	const { values, positionals } = parseArgs({
		allowPositionals: true,
		options: { limit: { type: "string" } },
	});

	const models = positionals;
	if (models.length === 0) {
		console.error("Usage: pnpm --filter @huggingface/hub dump-fixtures -- org/model [org/model…]");
		process.exit(1);
	}
	const limit = values.limit ? Number(values.limit) : DEFAULT_LIMIT;

	// The variants of the models are discovered, not listed by hand: they are the repos that
	// declare the model in their `base_model` metadata.
	const repos = new Set<string>();
	for (const model of models) {
		repos.add(model);
		for (const variant of await findVariantRepos(model, limit)) {
			repos.add(variant);
		}
	}

	await mkdir(join(FIXTURES_DIR, "models"), { recursive: true });
	await mkdir(join(FIXTURES_DIR, "configs"), { recursive: true });

	const captured: string[] = [];
	for (const repo of repos) {
		if (await dumpRepo(repo)) {
			captured.push(repo);
		}
	}

	// The fixtures of the repos that are gone are removed, so that the mock hub only serves the
	// repos the Hub serves.
	const capturedNames = new Set(captured.map((repo) => `${fixtureName(repo)}.json`));
	for (const dir of ["models", "configs"]) {
		for (const file of await readdir(join(FIXTURES_DIR, dir))) {
			if (file.endsWith(".json") && !capturedNames.has(file)) {
				await rm(join(FIXTURES_DIR, dir, file), { force: true });
			}
		}
	}

	console.log(`\n${captured.length} fixtures written to ${FIXTURES_DIR}`);

	// The order the Hub listed the repos in is kept, so that the menu of the mock hub is the
	// menu of the Hub, down to the order of the rows of the same size.
	await writeFile(join(FIXTURES_DIR, "manifest.json"), JSON.stringify({ models: captured }, null, "\t"), "utf8");
}

main();
