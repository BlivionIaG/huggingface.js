/**
 * Dumps the responses of the Hub API that the quantization menu needs, to the fixtures directory.
 *
 * The fixtures are served by `mock-hub.ts`, which lets the menu be tested offline, and without
 * hitting the Hub.
 *
 * Usage:
 *   pnpm --filter @huggingface/hub exec tsx scripts/dump-fixtures.ts
 *   pnpm --filter @huggingface/hub exec tsx scripts/dump-fixtures.ts Qwen/Qwen3.8-27B
 */

import { mkdir, readdir, rm, writeFile } from "node:fs/promises";
import { join } from "node:path";

const HUB_URL = process.env.HF_HUB_URL ?? "https://huggingface.co";
const FIXTURES_DIR = join(__dirname, "fixtures");

/**
 * Models the fixtures hold by default.
 *
 * They cover the three kinds of rows of the menu: a model that is not quantized (the base
 * model), a quantized repo, and a GGUF repo with its quantizations.
 */
const DEFAULT_MODELS = [
	"Qwen/Qwen3.8-27B",
	"Qwen/Qwen3.8-27B-FP8",
	"mlx-community/Qwen3.8-27B-4bit",
	"unsloth/Qwen3.8-27B-GGUF",
];

const models = process.argv.slice(2).length > 0 ? process.argv.slice(2) : DEFAULT_MODELS;

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

async function main() {
	await rm(join(FIXTURES_DIR, "models"), { recursive: true, force: true });
	await rm(join(FIXTURES_DIR, "configs"), { recursive: true, force: true });
	await mkdir(join(FIXTURES_DIR, "models"), { recursive: true });
	await mkdir(join(FIXTURES_DIR, "configs"), { recursive: true });

	for (const repo of models) {
		const infoResponse = await fetch(
			`${HUB_URL}/api/models/${repo}?blobs=true&expand[]=config&expand[]=safetensors&expand[]=cardData&expand[]=gguf`,
		);
		if (!infoResponse.ok) {
			throw new Error(`Could not fetch ${repo}: ${infoResponse.status} ${infoResponse.statusText}`);
		}
		const info = await infoResponse.json();
		await writeFile(
			join(FIXTURES_DIR, "models", `${fixtureName(repo)}.json`),
			JSON.stringify(pruneModelInfo(info), null, "\t"),
			"utf8",
		);

		// The raw config holds the dimensions of the model, which the `config` expansion does not.
		const configResponse = await fetch(`${HUB_URL}/${repo}/raw/main/config.json`);
		const config = configResponse.ok ? await configResponse.json() : undefined;
		await writeFile(
			join(FIXTURES_DIR, "configs", `${fixtureName(repo)}.json`),
			JSON.stringify({ id: repo, config }, null, "\t"),
			"utf8",
		);

		console.log(`${repo}: ${info.siblings?.length ?? 0} files`);
	}

	const files = await readdir(join(FIXTURES_DIR, "models"));
	console.log(`\n${files.length} fixtures written to ${FIXTURES_DIR}`);
}

main();
