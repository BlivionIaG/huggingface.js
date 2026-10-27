import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { startMockSite, type MockSite } from "./mock-site.js";

describe("mock site", () => {
	let site: MockSite;

	beforeAll(async () => {
		site = await startMockSite({ port: 0 });
	});

	afterAll(async () => {
		await site.close();
	});

	it("lists the models of the fixtures", async () => {
		const response = await fetch(`${site.url}/`);
		const html = await response.text();

		expect(response.status).toBe(200);
		expect(html).toContain("Qwen/Qwen3.8-27B");
		expect(html).toContain("unsloth/Qwen3.8-27B-GGUF");
	});

	it("shows the quantization menu of a model", async () => {
		const response = await fetch(`${site.url}/Qwen/Qwen3.8-27B`);
		const html = await response.text();

		expect(html).toContain("Quantizations");
		expect(html).toContain("FP8");
		expect(html).toContain("BF16");
	});

	it("marks the variants that fit in a hardware", async () => {
		const response = await fetch(`${site.url}/Qwen/Qwen3.8-27B?hardware=NVIDIA%20H100`);
		const html = await response.text();

		expect(html).toContain("chip fits");
		expect(html).toContain("chip tight");
	});

	it("shows one row per quantization of a gguf repo", async () => {
		const response = await fetch(`${site.url}/unsloth/Qwen3.8-27B-GGUF`);
		const html = await response.text();

		expect(html).toContain("UD-Q4_K_M");
		expect(html).toContain("UD-Q8_K_XL");
	});

	it("answers 404 for a model that does not exist", async () => {
		const response = await fetch(`${site.url}/does/not/exist`);

		expect(response.status).toBe(404);
	});
});
