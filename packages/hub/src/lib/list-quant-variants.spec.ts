import { describe, expect, it } from "vitest";
import { listQuantVariants } from "./list-quant-variants";

describe("listQuantVariants", () => {
	it("should list the base model and its quantized variants", async () => {
		const rows = await listQuantVariants({ name: "Qwen/Qwen3.8-27B" });

		const base = rows.find((row) => row.id === "Qwen/Qwen3.8-27B");
		expect(base).toBeDefined();
		expect(base?.label).toBe("BF16");
		expect(base?.bits).toBe(16);
		// 27.8B parameters in BF16
		expect(base?.sizeBytes).toBeGreaterThan(50e9);
		expect(base?.sizeBytes).toBeLessThan(60e9);

		const fp8 = rows.find((row) => row.id === "Qwen/Qwen3.8-27B-FP8");
		expect(fp8).toBeDefined();
		expect(fp8?.label).toBe("FP8");
		expect(fp8?.bits).toBe(8);
		// 27.8B parameters in FP8
		expect(fp8?.sizeBytes).toBeGreaterThan(28e9);
		expect(fp8?.sizeBytes).toBeLessThan(32e9);
	});

	it("should order the rows by quantization size", async () => {
		const rows = await listQuantVariants({ name: "Qwen/Qwen3.8-27B" });

		const bits = rows.map((row) => row.bits ?? 16);
		expect([...bits].sort((a, b) => a - b)).toEqual(bits);
	});

	it("should share the kv cache of the model between the variants", async () => {
		const rows = await listQuantVariants({ name: "Qwen/Qwen3.8-27B" });

		const base = rows.find((row) => row.id === "Qwen/Qwen3.8-27B");
		expect(base?.contextLength).toBe(262144);
		// 16 full attention layers, 4 kv heads, 256 head dim, 2 bytes, K and V
		expect(base?.kvCache?.bytesPerToken).toBe(65536);
	});

	it("should only list the variants of the author of the model by default", async () => {
		const rows = await listQuantVariants({ name: "Qwen/Qwen3.8-27B" });

		expect(rows.length).toBeGreaterThan(1);
		for (const row of rows) {
			expect(row.id.startsWith("Qwen/")).toBe(true);
		}
	});

	it("should list the gguf quantizations of a gguf repo", async () => {
		const rows = await listQuantVariants({ name: "unsloth/Qwen3.8-27B-GGUF" });

		const q4 = rows.find((row) => row.label === "Q4_0");
		expect(q4).toBeDefined();
		expect(q4?.file).toBe("Qwen3.8-27B-Q4_0.gguf");
		expect(q4?.sizeBytes).toBeGreaterThan(15e9);
		expect(q4?.bits).toBe(4);
	});
});
