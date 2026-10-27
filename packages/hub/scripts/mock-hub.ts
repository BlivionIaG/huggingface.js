/**
 * Serves the fixtures of the Hub API, so the quantization menu can be tested offline.
 *
 * The endpoints served are the ones the menu reads:
 *
 * - `GET /api/models?filter=base_model:<model>&author=<author>&expand[]=…` lists the variants of
 *   a model, the way the Hub does: the repos that declare the model in their `base_model`
 *   metadata.
 * - `GET /api/models/<repo>?blobs=true&expand[]=…` returns the info of a repo, with the size of
 *   its files.
 * - `GET /<repo>/raw/main/config.json` returns the config of a repo.
 *
 * Every request is logged, which shows what the client asks for.
 *
 * Usage:
 *   pnpm --filter @huggingface/hub mock-hub
 *   pnpm --filter @huggingface/hub quant-variants -- org/model --hub-url http://localhost:8787
 */

import { createServer } from "node:http";
import { readdir, readFile } from "node:fs/promises";
import { join } from "node:path";

const FIXTURES_DIR = join(__dirname, "fixtures");
const PORT = Number(process.env.PORT ?? 8787);

interface ModelInfo {
	id: string;
	author?: string;
	cardData?: { base_model?: string | string[] };
	safetensors?: { parameters: Record<string, number>; total: number };
	siblings: Array<{ rfilename: string; size?: number; blobId?: string }>;
	tags?: string[];
	[key: string]: unknown;
}

async function loadFixtures(fixturesDir?: string): Promise<{ models: ModelInfo[]; configs: Map<string, unknown> }> {
	const root = fixturesDir ?? FIXTURES_DIR;
	const models: ModelInfo[] = [];
	const configs = new Map<string, unknown>();

	for (const dir of ["models", "extra"]) {
		let files: string[] = [];
		try {
			files = await readdir(join(root, dir));
		} catch {
			continue;
		}
		for (const file of files) {
			if (!file.endsWith(".json")) {
				continue;
			}
			models.push(JSON.parse(await readFile(join(root, dir, file), "utf8")));
		}
	}

	for (const dir of ["configs", "extra"]) {
		let files: string[] = [];
		try {
			files = await readdir(join(root, dir));
		} catch {
			continue;
		}
		for (const file of files) {
			if (!file.endsWith(".json")) {
				continue;
			}
			const { id, config } = JSON.parse(await readFile(join(root, dir, file), "utf8"));
			configs.set(id, config);
		}
	}

	// The fixtures are ordered the way the Hub listed the repos in, so that the menu of the mock
	// hub is the menu of the Hub, down to the order of the rows of the same size.
	const order = await loadOrder(root);
	const position = new Map(order.map((id, index) => [id, index]));
	models.sort((a, b) => (position.get(a.id) ?? order.length) - (position.get(b.id) ?? order.length));

	return { models, configs };
}

/**
 * The order the Hub listed the repos in, as the dumper captured it.
 */
async function loadOrder(root: string): Promise<string[]> {
	try {
		const manifest = JSON.parse(await readFile(join(root, "manifest.json"), "utf8")) as { models?: string[] };
		return manifest.models ?? [];
	} catch {
		return [];
	}
}

/**
 * Whether a model declares a base model, the way the Hub links a quantization to the model it
 * comes from.
 */
function hasBaseModel(model: ModelInfo, baseModel: string): boolean {
	const base = model.cardData?.base_model;
	if (typeof base === "string") {
		return base === baseModel;
	}
	return Array.isArray(base) && base.includes(baseModel);
}

function getAuthor(model: ModelInfo): string {
	return model.author ?? model.id.split("/")[0];
}

/**
 * The model the fixtures hold the variants of, if there is one.
 *
 * A variant declares the model it comes from in its `base_model` metadata, so the base model is
 * the one the menu of the mock hub is about.
 */
function baseModelOf(models: ModelInfo[]): string | undefined {
	const bases = new Set<string>();
	for (const model of models) {
		const base = model.cardData?.base_model;
		for (const id of typeof base === "string" ? [base] : (base ?? [])) {
			bases.add(id);
		}
	}
	return bases.size === 1 ? [...bases][0] : undefined;
}

function listModels(models: ModelInfo[], url: URL): { models: Array<Record<string, unknown>>; total: number } {
	const filters = url.searchParams.getAll("filter");
	const authors = url.searchParams.get("author")?.split(",").filter(Boolean);
	const search = url.searchParams.get("search")?.toLowerCase();
	const limit = Number(url.searchParams.get("limit") ?? 20);
	const expand = url.searchParams.getAll("expand[]");

	let filtered = models.filter((model) => {
		for (const filter of filters) {
			if (filter.startsWith("base_model:")) {
				if (!hasBaseModel(model, filter.slice("base_model:".length))) {
					return false;
				}
			} else if (!model.tags?.includes(filter)) {
				return false;
			}
		}
		if (authors && !authors.includes(getAuthor(model))) {
			return false;
		}
		if (search && !model.id.toLowerCase().includes(search)) {
			return false;
		}
		return true;
	});

	const total = filtered.length;
	filtered = filtered.slice(0, limit);

	return {
		models: filtered.map((model) => ({
			_id: model._id,
			id: model.id,
			author: getAuthor(model),
			downloads: model.downloads,
			likes: model.likes,
			tags: model.tags,
			private: model.private,
			gated: model.gated,
			pipeline_tag: model.pipeline_tag,
			lastModified: model.lastModified,
			createdAt: model.createdAt,
			trendingScore: model.trendingScore,
			...(expand.includes("safetensors") ? { safetensors: model.safetensors } : {}),
			...(expand.includes("cardData") ? { cardData: model.cardData } : {}),
		})),
		total,
	};
}

/**
 * A mock hub, serving the fixtures.
 */
export interface MockHub {
	/**
	 * Url of the mock hub.
	 */
	url: string;
	/**
	 * Stops the mock hub.
	 */
	close: () => Promise<void>;
}

/**
 * Starts a mock hub, serving the fixtures of the directory.
 *
 * @param options port to listen on, 0 for a random one, and the directory of the fixtures
 */
export async function startMockHub(options?: { port?: number; fixturesDir?: string }): Promise<MockHub> {
	const { models, configs } = await loadFixtures(options?.fixturesDir);

	const server = createServer((req, res) => {
		const url = new URL(req.url ?? "/", "http://localhost");
		const path = decodeURIComponent(url.pathname);

		const send = (status: number, body: unknown) => {
			const payload = JSON.stringify(body);
			res.writeHead(status, { "content-type": "application/json" });
			res.end(payload);
		};

		// `GET /api/models`
		if (path === "/api/models") {
			const { models: list, total } = listModels(models, url);
			console.log(`→ GET ${url.pathname}${url.search}  (${list.length}/${total} models)`);
			send(200, list);
			return;
		}

		// `GET /api/models/<repo>`
		if (path.startsWith("/api/models/")) {
			const id = path.slice("/api/models/".length);
			const model = models.find((candidate) => candidate.id === id);
			if (!model) {
				console.log(`→ GET ${path}  (404)`);
				send(404, { error: `Model not found: ${id}` });
				return;
			}

			// The Hub only returns the size of the files when they are asked for.
			const blobs = url.searchParams.get("blobs") === "true";
			const siblings = blobs ? model.siblings : model.siblings.map((file) => ({ rfilename: file.rfilename }));

			console.log(`→ GET ${path}${url.search}  (${siblings.length} files)`);
			send(200, { ...model, siblings });
			return;
		}

		// `GET /<repo>/raw/main/config.json`
		const configMatch = path.match(/^\/?(.+)\/raw\/main\/config\.json$/);
		if (configMatch) {
			const id = configMatch[1];
			if (!configs.has(id)) {
				console.log(`→ GET ${path}  (404)`);
				send(404, { error: `Config not found: ${id}` });
				return;
			}
			console.log(`→ GET ${path}`);
			send(200, configs.get(id));
			return;
		}

		console.log(`→ GET ${path}  (404)`);
		send(404, { error: `Not found: ${path}` });
	});

	const port = options?.port ?? PORT;
	await new Promise<void>((resolve) => server.listen(port, resolve));

	const address = server.address();
	const actualPort = typeof address === "object" && address ? address.port : port;

	return {
		url: `http://localhost:${actualPort}`,
		close: () => new Promise<void>((resolve, reject) => server.close((error) => (error ? reject(error) : resolve()))),
	};
}

async function main() {
	const hub = await startMockHub();
	const { models } = await loadFixtures();

	console.log(`Mock hub listening on ${hub.url}`);
	console.log(`Serving ${models.length} models: ${models.map((model) => model.id).join(", ")}`);
	console.log("\nTry:");
	console.log(
		`  pnpm --filter @huggingface/hub quant-variants -- ${baseModelOf(models) ?? "<model>"} --hub-url ${hub.url}`,
	);
}

if (require.main === module) {
	void main();
}
