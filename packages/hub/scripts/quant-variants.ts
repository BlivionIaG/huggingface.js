/**
 * Prints the quantization menu of a model, the way the model page shows it.
 *
 * The menu of a GGUF repo lists the quantizations of the repo, one per `.gguf` file. This
 * script lists the same menu for any model: the base model, and the quantized variants that
 * were published for it (`Qwen/Qwen3.8-27B-FP8` and friends).
 *
 * The menu is read from the Hub. To read it from the local mock hub instead, start it with
 * `pnpm --filter @huggingface/hub mock-hub` and pass `--hub-url http://localhost:8787`.
 *
 * Usage:
 *   pnpm --filter @huggingface/hub exec tsx scripts/quant-variants.ts org/model
 *   pnpm --filter @huggingface/hub exec tsx scripts/quant-variants.ts org/model --hardware "RTX 4090"
 *   pnpm --filter @huggingface/hub exec tsx scripts/quant-variants.ts org/model --html menu.html
 *   pnpm --filter @huggingface/hub exec tsx scripts/quant-variants.ts org/model --hub-url http://localhost:8787
 */

import {
	SKUS,
	computeFit,
	formatBytes,
	getQuantVariantMemoryRequirement,
	type QuantVariantRow,
} from "@huggingface/tasks";
import { parseArgs } from "node:util";
import { writeFile } from "node:fs/promises";
import { listQuantVariants } from "../src/lib/list-quant-variants.js";
import { stripArgumentSeparator } from "./args.js";

stripArgumentSeparator();

const { values, positionals } = parseArgs({
	allowPositionals: true,
	options: {
		hardware: { type: "string" },
		memory: { type: "string" },
		context: { type: "string" },
		all: { type: "boolean" },
		limit: { type: "string" },
		"hub-url": { type: "string" },
		html: { type: "string" },
	},
});

const model = positionals[0];
if (!model) {
	console.error(
		"Usage: pnpm --filter @huggingface/hub quant-variants -- <model> [--hardware <name>] [--memory <GB>] [--context <tokens>] [--all] [--limit <n>] [--hub-url <url>] [--html <file>]",
	);
	process.exit(1);
}
const contextLength = values.context ? Number(values.context) : undefined;

interface Hardware {
	name: string;
	memoryBytes: number;
}

function findHardware(name: string): Hardware | undefined {
	for (const vendors of Object.values(SKUS)) {
		for (const [vendor, skus] of Object.entries(vendors)) {
			for (const [sku, spec] of Object.entries(skus)) {
				if (sku.toLowerCase() === name.toLowerCase() || sku.toLowerCase().includes(name.toLowerCase())) {
					const memory = values.memory ? Number(values.memory) : spec.memory?.[0];
					if (!memory) {
						continue;
					}
					return { name: `${vendor} ${sku}`, memoryBytes: memory * 1e9 };
				}
			}
		}
	}
	return undefined;
}

async function main() {
	let hardware = values.hardware ? findHardware(values.hardware) : undefined;
	if (values.hardware && !hardware) {
		console.error(`Unknown hardware "${values.hardware}"`);
		process.exit(1);
	}
	if (values.memory && !hardware) {
		hardware = { name: "custom", memoryBytes: Number(values.memory) * 1e9 };
	}

	const rows = await listQuantVariants({
		name: model,
		authors: values.all ? [] : "same",
		limit: values.limit ? Number(values.limit) : undefined,
		hubUrl: values["hub-url"],
	});

	const groups = new Map<number, QuantVariantRow[]>();
	for (const row of rows) {
		const bits = row.bits ?? 16;
		groups.set(bits, [...(groups.get(bits) ?? []), row]);
	}

	const fitOf = (row: QuantVariantRow) => {
		const requirement = getQuantVariantMemoryRequirement(row, { contextLength });
		return hardware ? computeFit(requirement, hardware.memoryBytes) : undefined;
	};

	console.log(`\n${model}`);
	console.log("=".repeat(model.length));
	if (rows[0]?.totalParams) {
		console.log(`${(rows[0].totalParams / 1e9).toFixed(1)}B parameters`);
	}
	if (rows[0]?.contextLength) {
		console.log(`context length: ${rows[0].contextLength.toLocaleString()} tokens`);
	}
	if (hardware) {
		console.log(`hardware: ${hardware.name} (${formatBytes(hardware.memoryBytes)})`);
	}

	for (const [bits, group] of [...groups.entries()].sort((a, b) => a[0] - b[0])) {
		console.log(`\n${bits}-bit`);
		for (const row of group) {
			const fit = fitOf(row);
			const mark = fit === undefined ? " " : fit.state === "fits" ? "✅" : fit.state === "tight" ? "🟠" : "❌";
			const size = formatBytes(row.sizeBytes).padStart(9);
			const name = row.file ? `${row.id}/${row.file}` : row.id;
			console.log(`  ${mark} ${row.label.padEnd(10)} ${size}  ${name}`);
			if (fit) {
				const hint = fit.state === "tight" && fit.weightsFit ? " (fits with a shorter context)" : "";
				console.log(`      needs ${formatBytes(fit.requiredBytes)} of ${formatBytes(fit.availableBytes)}${hint}`);
			}
		}
	}

	if (values.html) {
		await writeFile(values.html, renderHtml(rows, { hardware, contextLength }), "utf8");
		console.log(`\nWidget mock written to ${values.html}`);
	}
}

function escapeHtml(value: string): string {
	return value.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

/**
 * Renders a mock of the quantization widget of a model page.
 */
function renderHtml(rows: QuantVariantRow[], options: { hardware?: Hardware; contextLength?: number }): string {
	const groups = new Map<number, QuantVariantRow[]>();
	for (const row of rows) {
		const bits = row.bits ?? 16;
		groups.set(bits, [...(groups.get(bits) ?? []), row]);
	}

	const groupHtml = [...groups.entries()]
		.sort((a, b) => a[0] - b[0])
		.map(([bits, group]) => {
			const chips = group
				.map((row) => {
					const requirement = getQuantVariantMemoryRequirement(row, {
						contextLength: options.contextLength,
					});
					const fit = options.hardware ? computeFit(requirement, options.hardware.memoryBytes) : undefined;
					const state = fit?.state ?? "unknown";
					const icon = state === "fits" ? "✅" : state === "tight" ? "🟠" : state === "no-fit" ? "❌" : "";
					return `<span class="chip ${state}">${icon} ${escapeHtml(row.label)} <span class="size">${formatBytes(
						row.sizeBytes,
					)}</span></span>`;
				})
				.join("\n");
			return `<div class="group"><div class="bits">${bits}-bit</div><div class="chips">${chips}</div></div>`;
		})
		.join("\n");

	return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>${escapeHtml(model)} — quantization menu</title>
<style>
	:root { color-scheme: light; }
	body { font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif; background: #f9fafb; color: #111827; margin: 0; padding: 32px; }
	.card { background: #fff; border: 1px solid #e5e7eb; border-radius: 12px; padding: 20px; max-width: 720px; box-shadow: 0 1px 2px rgba(0,0,0,.04); }
	h1 { font-size: 18px; margin: 0 0 4px; }
	.sub { color: #6b7280; font-size: 13px; margin-bottom: 16px; }
	.hardware { display: inline-flex; align-items: center; gap: 8px; border: 1px solid #e5e7eb; border-radius: 8px; padding: 8px 12px; font-size: 14px; margin-bottom: 16px; background: #fff; }
	.group { display: flex; gap: 16px; padding: 10px 0; border-top: 1px solid #f3f4f6; }
	.bits { width: 56px; flex: none; color: #6b7280; font-size: 13px; padding-top: 4px; }
	.chips { display: flex; flex-wrap: wrap; gap: 8px; }
	.chip { display: inline-flex; align-items: center; gap: 6px; border: 1px solid #e5e7eb; border-radius: 8px; padding: 5px 10px; font-size: 13px; background: #fff; }
	.chip.fits { border-color: #bbf7d0; background: #f0fdf4; }
	.chip.tight { border-color: #fed7aa; background: #fff7ed; }
	.chip.no-fit { border-color: #e5e7eb; background: #f9fafb; color: #9ca3af; }
	.size { color: #6b7280; font-size: 12px; }
</style>
</head>
<body>
	<div class="card">
		<h1>${escapeHtml(model)}</h1>
		<div class="sub">${
			rows[0]?.totalParams ? `${(rows[0].totalParams / 1e9).toFixed(1)}B parameters` : ""
		}${rows[0]?.contextLength ? ` · context ${rows[0].contextLength.toLocaleString()} tokens` : ""}</div>
		${
			options.hardware
				? `<div class="hardware">🖥️ ${escapeHtml(options.hardware.name)} (${formatBytes(
						options.hardware.memoryBytes,
					)})</div>`
				: ""
		}
		${groupHtml}
	</div>
</body>
</html>
`;
}

main();
