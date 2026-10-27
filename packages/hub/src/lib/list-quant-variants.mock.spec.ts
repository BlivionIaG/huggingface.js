import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { startMockHub, type MockHub } from "../../scripts/mock-hub";
import { listQuantVariants } from "./list-quant-variants";

/**
 * The menu, read from a mock hub.
 *
 * The fixtures of the mock hub are the responses of the real Hub, so the menu built here is the
 * one the model page shows, without hitting the network.
 */
describe("listQuantVariants (mock hub)", () => {
	let hub: MockHub;

	beforeAll(async () => {
		// Port 0: the mock hub listens on a random port.
		hub = await startMockHub({ port: 0 });
	});

	afterAll(async () => {
		await hub.close();
	});

	it("should list the base model and the variants of its author", async () => {
		const rows = await listQuantVariants({ name: "Qwen/Qwen3.8-27B", hubUrl: hub.url });

		const base = rows.find((row) => row.id === "Qwen/Qwen3.8-27B");
		expect(base?.label).toBe("BF16");
		expect(base?.sizeBytes).toBeGreaterThan(50e9);

		const fp8 = rows.find((row) => row.id === "Qwen/Qwen3.8-27B-FP8");
		expect(fp8?.label).toBe("FP8");
		expect(fp8?.bits).toBe(8);
		expect(fp8?.sizeBytes).toBeGreaterThan(28e9);
		expect(fp8?.sizeBytes).toBeLessThan(32e9);
	});

	it("should read the kv cache of the model from its config", async () => {
		const rows = await listQuantVariants({ name: "Qwen/Qwen3.8-27B", hubUrl: hub.url });

		const base = rows.find((row) => row.id === "Qwen/Qwen3.8-27B");
		expect(base?.contextLength).toBe(262144);
		expect(base?.kvCache?.bytesPerToken).toBe(65536);
	});

	it("should not list the adapters of the model", async () => {
		const rows = await listQuantVariants({ name: "Qwen/Qwen3.8-27B", authors: [], hubUrl: hub.url });

		// An adapter holds the weights of the layers it trains, not the model
		expect(rows.find((row) => row.id === "someone/Qwen3.8-27B-Adapter-4bit")).toBeUndefined();
	});

	it("should list the variants of every author", async () => {
		const rows = await listQuantVariants({ name: "Qwen/Qwen3.8-27B", authors: [], hubUrl: hub.url });

		const ids = rows.map((row) => row.id);
		expect(ids).toContain("Qwen/Qwen3.8-27B-FP8");
		expect(ids).toContain("mlx-community/Qwen3.8-27B-4bit");
		expect(ids).toContain("unsloth/Qwen3.8-27B-GGUF");
	});

	it("should list one row per quantization of a gguf repo", async () => {
		const rows = await listQuantVariants({ name: "unsloth/Qwen3.8-27B-GGUF", hubUrl: hub.url });

		const q4 = rows.find((row) => row.label === "Q4_0");
		expect(q4?.file).toBe("Qwen3.8-27B-Q4_0.gguf");
		expect(q4?.sizeBytes).toBe(16056478688);
		expect(q4?.bits).toBe(4);

		// The two shards of the BF16 release are a single row
		const bf16 = rows.find((row) => row.label === "BF16");
		expect(bf16?.files).toHaveLength(2);
		expect(bf16?.sizeBytes).toBe(54657735616);
	});
});
