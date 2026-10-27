/**
 * The quantization menu of a model page, rendered as HTML.
 *
 * The menu is the one the model page shows: the quantizations of the model, grouped by the number
 * of bits per weight, with the size of each and whether it fits in a hardware.
 */

import { computeFit, formatBytes, getQuantVariantMemoryRequirement, type QuantVariantRow } from "@huggingface/tasks";

/**
 * A hardware the menu is compared to.
 */
export interface Hardware {
	name: string;
	memoryBytes: number;
}

/**
 * Options of the menu.
 */
export interface MenuOptions {
	/**
	 * Hardware the variants are compared to. Without it, the menu shows the sizes only.
	 */
	hardware?: Hardware;
	/**
	 * Context length of the model, when it is not the one of the model.
	 */
	contextLength?: number;
}

/**
 * Escapes a value for the HTML.
 */
export function escapeHtml(value: string): string {
	return value.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

/**
 * The chips of the menu, grouped by the number of bits per weight.
 */
export function renderQuantizationMenu(rows: QuantVariantRow[], options: MenuOptions): string {
	const groups = new Map<number, QuantVariantRow[]>();
	for (const row of rows) {
		const bits = row.bits ?? 16;
		groups.set(bits, [...(groups.get(bits) ?? []), row]);
	}

	return [...groups.entries()]
		.sort((a, b) => a[0] - b[0])
		.map(([bits, group]) => {
			const chips = group
				.map((row) => {
					const requirement = getQuantVariantMemoryRequirement(row, { contextLength: options.contextLength });
					const fit = options.hardware ? computeFit(requirement, options.hardware.memoryBytes) : undefined;
					const state = fit?.state ?? "unknown";
					const icon = state === "fits" ? "✅" : state === "tight" ? "🟠" : state === "no-fit" ? "❌" : "";
					const title = fit
						? `needs ${formatBytes(fit.requiredBytes)} of ${formatBytes(fit.availableBytes)}`
						: formatBytes(row.sizeBytes);
					const name = row.file ? `${row.id}/${row.file}` : row.id;
					return `<a class="chip ${state}" href="/${escapeHtml(row.id)}" title="${escapeHtml(name)} — ${escapeHtml(
						title,
					)}">${icon} ${escapeHtml(row.label)} <span class="size">${formatBytes(row.sizeBytes)}</span></a>`;
				})
				.join("\n");
			return `<div class="group"><div class="bits">${bits}-bit</div><div class="chips">${chips}</div></div>`;
		})
		.join("\n");
}

/**
 * The styles of the menu.
 */
export const MENU_STYLES = `
	.menu { display: flex; flex-direction: column; gap: 2px; }
	.menu .group { display: flex; gap: 12px; padding: 6px 0; border-top: 1px solid #f3f4f6; }
	.menu .group:first-child { border-top: none; }
	.menu .bits { width: 48px; flex: none; color: #6b7280; font-size: 12px; padding-top: 5px; }
	.menu .chips { display: flex; flex-wrap: wrap; gap: 6px; }
	.chip { display: inline-flex; align-items: center; gap: 5px; border: 1px solid #e5e7eb; border-radius: 6px; padding: 3px 8px; font-size: 12px; background: #fff; color: #111827; text-decoration: none; }
	.chip:hover { border-color: #f59e0b; }
	.chip .size { color: #9ca3af; }
	.chip.fits { border-color: #bbf7d0; background: #f0fdf4; }
	.chip.tight { border-color: #fed7aa; background: #fff7ed; }
	.chip.no-fit { background: #f9fafb; color: #9ca3af; }
	.chip.unknown { border-style: dashed; }
`;
