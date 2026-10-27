/**
 * Serves a mock of the Hugging Face website, so the quantization menu of a model page can be seen
 * without the network.
 *
 * The pages are built from the API of the mock hub, the way the pages of the Hub are built from
 * the Hub API: the menu of a model page is the menu of `listQuantVariants`, read from the API.
 *
 * The pages served are:
 *
 * - `GET /` lists the models of the fixtures.
 * - `GET /<owner>/<model>` shows a model page, with the quantization menu of the model.
 *
 * The hardware the menu is compared to is chosen with the `hardware` query parameter, and the
 * memory of a custom hardware with the `memory` one.
 *
 * Usage:
 *   pnpm --filter @huggingface/hub mock-site
 *   pnpm --filter @huggingface/hub mock-site -- --port 8080
 */

import { createServer } from "node:http";
import { SKUS, type HardwareSpec } from "@huggingface/tasks";
import { listQuantVariants } from "../src/lib/list-quant-variants.js";
import { stripArgumentSeparator } from "./args.js";
import { loadFixtures, startMockHub, type ModelInfo } from "./mock-hub.js";
import { escapeHtml, MENU_STYLES, renderQuantizationMenu, type Hardware } from "./widget.js";

stripArgumentSeparator();

const PORT = Number(process.env.PORT ?? 8787);

/**
 * The styles of the pages, which look like the ones of the Hub.
 */
const SITE_STYLES = `
	:root { color-scheme: light; --yellow: #f59e0b; }
	* { box-sizing: border-box; }
	body { font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif; background: #fff; color: #111827; margin: 0; }
	a { color: inherit; text-decoration: none; }
	header.nav { display: flex; align-items: center; gap: 20px; padding: 10px 24px; border-bottom: 1px solid #e5e7eb; background: #fff; position: sticky; top: 0; z-index: 10; }
	header.nav .logo { display: flex; align-items: center; gap: 8px; font-weight: 700; font-size: 18px; }
	header.nav .logo .emoji { font-size: 22px; }
	header.nav .tabs { display: flex; gap: 18px; color: #6b7280; font-size: 14px; }
	header.nav .tabs a.active { color: #111827; font-weight: 600; }
	header.nav .search { margin-left: auto; }
	input[type="search"], input[type="number"], select { border: 1px solid #e5e7eb; border-radius: 8px; padding: 7px 10px; font-size: 13px; background: #f9fafb; color: #111827; }
	main { max-width: 1100px; margin: 0 auto; padding: 24px; }
	.repo { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
	.repo h1 { font-size: 22px; margin: 0; font-weight: 500; }
	.repo .owner { color: #6b7280; }
	.tags { display: flex; gap: 6px; flex-wrap: wrap; margin: 10px 0; }
	.tag { border: 1px solid #e5e7eb; border-radius: 999px; padding: 2px 9px; font-size: 12px; color: #4b5563; background: #f9fafb; }
	.stats { display: flex; gap: 20px; color: #6b7280; font-size: 13px; margin: 8px 0 16px; }
	.tabs { display: flex; gap: 18px; border-bottom: 1px solid #e5e7eb; margin-bottom: 18px; font-size: 14px; }
	.tabs a { padding: 8px 0; color: #6b7280; border-bottom: 2px solid transparent; }
	.tabs a.active { color: #111827; border-bottom-color: var(--yellow); font-weight: 600; }
	.card { background: #fff; border: 1px solid #e5e7eb; border-radius: 12px; padding: 18px; }
	.card h2 { font-size: 15px; margin: 0 0 4px; }
	.card .sub { color: #6b7280; font-size: 13px; margin-bottom: 12px; }
	.hardware { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; margin-bottom: 12px; font-size: 13px; }
	.hardware button { border: 1px solid #e5e7eb; border-radius: 8px; padding: 7px 12px; font-size: 13px; background: #111827; color: #fff; cursor: pointer; }
	table.models { width: 100%; border-collapse: collapse; font-size: 14px; }
	table.models th { text-align: left; color: #6b7280; font-weight: 500; font-size: 12px; text-transform: uppercase; padding: 8px 10px; border-bottom: 1px solid #e5e7eb; }
	table.models td { padding: 10px; border-bottom: 1px solid #f3f4f6; }
	table.models tr:hover td { background: #f9fafb; }
	.id { font-weight: 600; }
	.muted { color: #6b7280; }
	.empty { color: #9ca3af; font-size: 13px; padding: 10px 0; }
`;

/**
 * The hardware the menu can be compared to, from the SKUs of the library.
 */
function hardwareOptions(): Hardware[] {
	// The specs of the vendors do not have the same type, but they all extend the `HardwareSpec`.
	const groups = Object.values(SKUS) as Array<Record<string, Record<string, HardwareSpec>>>;
	const options: Hardware[] = [];
	for (const vendors of groups) {
		for (const [vendor, skus] of Object.entries(vendors)) {
			for (const [sku, spec] of Object.entries(skus)) {
				const memory = spec.memory?.[0];
				if (memory) {
					options.push({ name: `${vendor} ${sku}`, memoryBytes: memory * 1e9 });
				}
			}
		}
	}
	return options;
}

const HARDWARE = hardwareOptions();

/**
 * The hardware the menu is compared to, from the query of the page.
 */
function hardwareOf(url: URL): Hardware | undefined {
	const memory = url.searchParams.get("memory");
	if (memory) {
		const bytes = Number(memory) * 1e9;
		return Number.isFinite(bytes) && bytes > 0 ? { name: "custom", memoryBytes: bytes } : undefined;
	}

	const name = url.searchParams.get("hardware");
	if (!name) {
		return undefined;
	}
	return HARDWARE.find((option) => option.name === name);
}

function renderPage(title: string, body: string): string {
	return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>${escapeHtml(title)}</title>
<style>${SITE_STYLES}${MENU_STYLES}</style>
</head>
<body>
<header class="nav">
	<a class="logo" href="/"><span class="emoji">🤗</span> Hugging Face</a>
	<nav class="tabs">
		<a class="active" href="/">Models</a>
		<a href="/">Datasets</a>
		<a href="/">Spaces</a>
		<a href="/">Docs</a>
	</nav>
	<div class="search"><input type="search" placeholder="Search models, datasets, spaces…" size="34"></div>
</header>
<main>${body}</main>
</body>
</html>
`;
}

/**
 * The list of the models of the fixtures.
 */
function renderHome(models: ModelInfo[]): string {
	const rows = models
		.map(
			(model) => `<tr>
				<td><a class="id" href="/${escapeHtml(model.id)}">${escapeHtml(model.id)}</a></td>
				<td class="muted">${escapeHtml(String(model.pipeline_tag ?? ""))}</td>
				<td class="muted">${(model.downloads ?? 0).toLocaleString()}</td>
				<td class="muted">${(model.likes ?? 0).toLocaleString()}</td>
			</tr>`,
		)
		.join("\n");

	return renderPage(
		"Models · Hugging Face",
		`<table class="models">
			<thead><tr><th>Model</th><th>Task</th><th>Downloads</th><th>Likes</th></tr></thead>
			<tbody>${rows}</tbody>
		</table>`,
	);
}

/**
 * The page of a model, with the quantization menu of the model.
 */
async function renderModelPage(model: ModelInfo, url: URL, hubUrl: string): Promise<string> {
	const hardware = hardwareOf(url);
	const rows = await listQuantVariants({ name: model.id, hubUrl });

	const menu =
		rows.length === 0
			? `<div class="empty">No quantization is published for this model.</div>`
			: `<div class="menu">${renderQuantizationMenu(rows, { hardware })}</div>`;

	const options = HARDWARE.map(
		(option) =>
			`<option value="${escapeHtml(option.name)}"${hardware?.name === option.name ? " selected" : ""}>${escapeHtml(
				option.name,
			)}</option>`,
	).join("");

	const chips = (model.tags ?? [])
		.slice(0, 8)
		.map((tag) => `<span class="tag">${escapeHtml(tag)}</span>`)
		.join("");

	return renderPage(
		`${model.id} · Hugging Face`,
		`<div class="repo">
			<h1><span class="owner">${escapeHtml(model.author ?? model.id.split("/")[0])}/</span>${escapeHtml(
				model.id.split("/")[1] ?? "",
			)}</h1>
		</div>
		<div class="tags">${chips}</div>
		<div class="stats">
			<span>Downloads last month: ${(model.downloads ?? 0).toLocaleString()}</span>
			<span>Likes: ${(model.likes ?? 0).toLocaleString()}</span>
			${rows[0]?.totalParams ? `<span>${(rows[0].totalParams / 1e9).toFixed(1)}B parameters</span>` : ""}
			${rows[0]?.contextLength ? `<span>Context: ${rows[0].contextLength.toLocaleString()} tokens</span>` : ""}
		</div>
		<nav class="tabs">
			<a class="active" href="/${escapeHtml(model.id)}">README</a>
			<a href="/${escapeHtml(model.id)}">Files and versions</a>
			<a href="/${escapeHtml(model.id)}">Community</a>
		</nav>
		<div class="card">
			<h2>Quantizations</h2>
			<div class="sub">The quantizations published for this model, and the memory they need.</div>
			<form class="hardware" method="get" action="/${escapeHtml(model.id)}">
				<label for="hardware">Hardware</label>
				<select id="hardware" name="hardware">
					<option value=""${hardware ? "" : " selected"}>None</option>
					${options}
				</select>
				<input type="number" name="memory" placeholder="GB" min="1" step="1" style="width:90px" value="${
					hardware?.name === "custom" ? (hardware.memoryBytes / 1e9).toFixed(0) : ""
				}">
				<button type="submit">Apply</button>
			</form>
			${menu}
		</div>`,
	);
}

/**
 * A mock site, serving the pages of the Hub.
 */
export interface MockSite {
	/**
	 * Url of the mock site.
	 */
	url: string;
	/**
	 * Stops the mock site.
	 */
	close: () => Promise<void>;
}

/**
 * Starts a mock site, serving the fixtures of the mock hub.
 *
 * @param options port to listen on, 0 for a random one
 */
export async function startMockSite(options?: { port?: number }): Promise<MockSite> {
	// The API of the mock hub, the pages are built from it.
	const hub = await startMockHub({ port: 0 });
	const { models } = await loadFixtures();

	const server = createServer((req, res) => {
		const url = new URL(req.url ?? "/", "http://localhost");
		const path = decodeURIComponent(url.pathname);

		const send = (status: number, body: string) => {
			res.writeHead(status, { "content-type": "text/html; charset=utf-8" });
			res.end(body);
		};

		void (async () => {
			if (path === "/" || path === "/models") {
				send(200, renderHome(models));
				return;
			}

			const model = models.find((candidate) => candidate.id === path.slice(1));
			if (model) {
				send(200, await renderModelPage(model, url, hub.url));
				return;
			}

			send(404, renderPage("Not found", `<div class="empty">No model at ${escapeHtml(path)}.</div>`));
		})();
	});

	const port = options?.port ?? PORT;
	await new Promise<void>((resolve) => server.listen(port, resolve));
	const address = server.address();
	const actualPort = typeof address === "object" && address ? address.port : port;

	return {
		url: `http://localhost:${actualPort}`,
		close: async () => {
			await new Promise<void>((resolve, reject) => server.close((error) => (error ? reject(error) : resolve())));
			await hub.close();
		},
	};
}

async function main() {
	const args = process.argv.slice(2);
	const portIndex = args.indexOf("--port");
	const port = portIndex === -1 ? PORT : Number(args[portIndex + 1]);

	const site = await startMockSite({ port });

	console.log(`Mock site listening on ${site.url}`);
	console.log("\nTry:");
	console.log(`  ${site.url}/Qwen/Qwen3.8-27B`);
	console.log(`  ${site.url}/Qwen/Qwen3.8-27B?hardware=NVIDIA%20RTX%204090`);
}

if (require.main === module) {
	void main();
}
