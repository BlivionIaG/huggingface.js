import { describe, expect, it } from "vitest";
import { parseGGUFQuantLabel } from "./gguf.js";

describe("parseGGUFQuantLabel", () => {
	it("should read the quantization of a file name", () => {
		expect(parseGGUFQuantLabel("Qwen3.8-27B-Q4_K_M.gguf")).toBe("Q4_K_M");
		expect(parseGGUFQuantLabel("Qwen3.8-27B-Q8_0.gguf")).toBe("Q8_0");
		expect(parseGGUFQuantLabel("Qwen3.8-27B-BF16.gguf")).toBe("BF16");
	});

	it("should read the vendor prefix and the size variation", () => {
		expect(parseGGUFQuantLabel("Qwen3.8-27B-UD-IQ1_S.gguf")).toBe("UD-IQ1_S");
		expect(parseGGUFQuantLabel("Qwen3.8-27B-UD-Q8_K_XL.gguf")).toBe("UD-Q8_K_XL");
		expect(parseGGUFQuantLabel("Qwen3.8-27B-UD-Q8_K_L.gguf")).toBe("UD-Q8_K_L");
	});

	it("should prefer the last quantization of the name", () => {
		expect(parseGGUFQuantLabel("Qwen3.8-27B-UD-Q4_K_M.gguf")).toBe("UD-Q4_K_M");
	});
});
