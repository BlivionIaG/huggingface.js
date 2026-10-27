import { describe, expect, it } from "vitest";
import {
	attentionConfigFromModelConfig,
	computeFit,
	describeBaseVariant,
	estimateMemoryRequirement,
	formatBytes,
	getGGUFQuantBits,
	getGGUFVariantRows,
	getKvCacheBytes,
	getKvCacheSpec,
	getQuantVariantMemoryRequirement,
	isWeightFile,
	kvCacheBytesPerToken,
	parseQuantVariantDescriptor,
	sumWeightFileSizes,
} from "./quant-variants.js";

describe("parseQuantVariantDescriptor", () => {
	it("should detect the format of a variant from its name", () => {
		expect(parseQuantVariantDescriptor("Qwen/Qwen3.8-27B-FP8")).toEqual({
			label: "FP8",
			format: "fp8",
			bits: 8,
		});
	});

	it("should detect a scheme and a bit width", () => {
		expect(parseQuantVariantDescriptor("cyankiwi/Qwen3.8-27B-AWQ-INT4")).toEqual({
			label: "AWQ INT4",
			format: "awq",
			bits: 4,
		});
		expect(parseQuantVariantDescriptor("someuser/Qwen3.8-27B-GPTQ-Int4")).toEqual({
			label: "GPTQ INT4",
			format: "gptq",
			bits: 4,
		});
	});

	it("should detect the microscaling formats before the plain ones", () => {
		// `NVFP4` contains `FP4` and `MXFP8` contains `FP8`
		expect(parseQuantVariantDescriptor("nvidia/Qwen3.8-27B-NVFP4")).toEqual({
			label: "NVFP4",
			format: "nvfp4",
			bits: 4,
		});
		expect(parseQuantVariantDescriptor("mlx-community/Qwen3.8-27B-mxfp8")).toEqual({
			label: "MXFP8",
			format: "mxfp8",
			bits: 8,
		});
	});

	it("should detect the MLX variants", () => {
		expect(parseQuantVariantDescriptor("mlx-community/Qwen3.8-27B-4bit")).toEqual({
			label: "MLX 4bit",
			format: "mlx",
			bits: 4,
		});
	});

	it("should not detect a format on a plain model", () => {
		expect(parseQuantVariantDescriptor("Qwen/Qwen3.8-27B")).toBeUndefined();
		expect(parseQuantVariantDescriptor("Qwen/Qwen3.8-27B-Instruct")).toBeUndefined();
	});

	it("should not detect a format on the size of a model", () => {
		// `8B` is a number of parameters, not a quantization
		expect(parseQuantVariantDescriptor("meta-llama/Llama-3.1-8B-Instruct")).toBeUndefined();
		expect(parseQuantVariantDescriptor("meta-llama/Llama-2-13b-chat-hf")).toBeUndefined();
	});
});

describe("describeBaseVariant", () => {
	it("should describe the dtype of the weights", () => {
		expect(describeBaseVariant("BF16")).toEqual({ label: "BF16", format: "bf16", bits: 16 });
		expect(describeBaseVariant("float16")).toEqual({ label: "FP16", format: "fp16", bits: 16 });
		expect(describeBaseVariant("F32")).toEqual({ label: "FP32", format: "fp32", bits: 32 });
	});

	it("should return undefined for an unknown dtype", () => {
		expect(describeBaseVariant("I8")).toBeUndefined();
		expect(describeBaseVariant(undefined)).toBeUndefined();
	});
});

describe("isWeightFile", () => {
	it("should detect the files that hold the weights", () => {
		expect(isWeightFile("model.safetensors")).toBe(true);
		expect(isWeightFile("Qwen3.8-27B-Q4_K_M.gguf")).toBe(true);
		expect(isWeightFile("pytorch_model.bin")).toBe(true);
	});

	it("should not detect the files that do not hold the weights", () => {
		expect(isWeightFile("model.safetensors.index.json")).toBe(false);
		expect(isWeightFile("config.json")).toBe(false);
		expect(isWeightFile("tokenizer.json")).toBe(false);
		expect(isWeightFile("README.md")).toBe(false);
	});
});

describe("sumWeightFileSizes", () => {
	it("should sum the size of the weight files only", () => {
		const { totalBytes, files } = sumWeightFileSizes([
			{ rfilename: "model.safetensors", size: 100 },
			{ rfilename: "model.safetensors.index.json", size: 10 },
			{ rfilename: "config.json", size: 5 },
			{ rfilename: "README.md", size: 1 },
		]);

		expect(totalBytes).toBe(100);
		expect(files).toEqual([{ path: "model.safetensors", sizeBytes: 100 }]);
	});

	it("should sum the shards of a same file", () => {
		const { totalBytes } = sumWeightFileSizes([
			{ rfilename: "model-00001-of-00002.safetensors", size: 100 },
			{ rfilename: "model-00002-of-00002.safetensors", size: 50 },
		]);

		expect(totalBytes).toBe(150);
	});

	it("should count one serialization of the weights only", () => {
		// `pytorch_model.bin` is another serialization of the same weights
		const { totalBytes } = sumWeightFileSizes([
			{ rfilename: "model.safetensors", size: 100 },
			{ rfilename: "pytorch_model.bin", size: 100 },
		]);

		expect(totalBytes).toBe(100);
	});

	it("should sum the shards of a file that is not named after the convention", () => {
		// Some repos split their weights in `layers-<n>.safetensors`
		const { totalBytes } = sumWeightFileSizes([
			{ rfilename: "layers-0.safetensors", size: 100 },
			{ rfilename: "layers-1.safetensors", size: 50 },
		]);

		expect(totalBytes).toBe(150);
	});

	it("should prefer the shards of a file over another serialization", () => {
		const { totalBytes } = sumWeightFileSizes([
			{ rfilename: "model.safetensors", size: 100 },
			{ rfilename: "model-00001-of-00002.safetensors", size: 60 },
			{ rfilename: "model-00002-of-00002.safetensors", size: 40 },
		]);

		expect(totalBytes).toBe(100);
	});
});

describe("getGGUFVariantRows", () => {
	it("should build one row per gguf file", () => {
		const rows = getGGUFVariantRows("unsloth/Qwen3.8-27B-GGUF", [
			{ rfilename: "Qwen3.8-27B-Q4_K_M.gguf", size: 16056478688 },
			{ rfilename: "Qwen3.8-27B-Q8_0.gguf", size: 29047086048 },
			{ rfilename: "README.md", size: 100 },
		]);

		expect(rows).toEqual([
			{
				id: "unsloth/Qwen3.8-27B-GGUF",
				file: "Qwen3.8-27B-Q4_K_M.gguf",
				files: ["Qwen3.8-27B-Q4_K_M.gguf"],
				label: "Q4_K_M",
				format: "gguf",
				sizeBytes: 16056478688,
				role: "model",
			},
			{
				id: "unsloth/Qwen3.8-27B-GGUF",
				file: "Qwen3.8-27B-Q8_0.gguf",
				files: ["Qwen3.8-27B-Q8_0.gguf"],
				label: "Q8_0",
				format: "gguf",
				sizeBytes: 29047086048,
				role: "model",
			},
		]);
	});

	it("should group the files of a quantization split in several shards", () => {
		const rows = getGGUFVariantRows("unsloth/Qwen3.8-27B-GGUF", [
			{ rfilename: "BF16/Qwen3.8-27B-BF16-00001-of-00002.gguf", size: 49986159616 },
			{ rfilename: "BF16/Qwen3.8-27B-BF16-00002-of-00002.gguf", size: 4671576000 },
		]);

		expect(rows).toHaveLength(1);
		expect(rows[0].label).toBe("BF16");
		expect(rows[0].file).toBeUndefined();
		expect(rows[0].files).toEqual([
			"BF16/Qwen3.8-27B-BF16-00001-of-00002.gguf",
			"BF16/Qwen3.8-27B-BF16-00002-of-00002.gguf",
		]);
		expect(rows[0].sizeBytes).toBe(54657735616);
	});

	it("should keep the files of two different models apart", () => {
		const rows = getGGUFVariantRows("huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF", [
			{ rfilename: "Huihui-Qwen3.8-27B-abliterated-UD-DW-bf16.gguf", size: 54657735616 },
			{ rfilename: "Huihui-Qwen3.8-27B-abliterated-bf16.gguf", size: 54657735616 },
		]);

		expect(rows).toHaveLength(2);
		expect(rows.map((row) => row.sizeBytes)).toEqual([54657735616, 54657735616]);
	});

	it("should tell the projector and the mtp head apart", () => {
		const rows = getGGUFVariantRows("unsloth/Qwen3.8-27B-GGUF", [
			{ rfilename: "mmproj-Qwen3.8-27B-F16.gguf", size: 100 },
			{ rfilename: "mtp-Qwen3.8-27B-Q4_0.gguf", size: 200 },
		]);

		expect(rows.map((row) => row.role)).toEqual(["mmproj", "mtp"]);
		expect(rows.map((row) => row.label)).toEqual(["mmproj F16", "MTP Q4_0"]);
	});

	it("should leave the quantization matrix out", () => {
		const rows = getGGUFVariantRows("unsloth/Qwen3.8-27B-GGUF", [
			{ rfilename: "imatrix_unsloth.gguf", size: 13600000 },
			{ rfilename: "Qwen3.8-27B-Q4_0.gguf", size: 16056478688 },
		]);

		expect(rows.map((row) => row.label)).toEqual(["Q4_0"]);
	});
});

describe("getGGUFQuantBits", () => {
	it("should return the number of bits of a quantization", () => {
		expect(getGGUFQuantBits("Q4_K_M")).toBe(4);
		expect(getGGUFQuantBits("Q8_0")).toBe(8);
		expect(getGGUFQuantBits("BF16")).toBe(16);
		expect(getGGUFQuantBits("UD-IQ1_S")).toBe(1);
		expect(getGGUFQuantBits("Q2_K_XL")).toBe(2);
		expect(getGGUFQuantBits("IQ2_XXS")).toBe(2);
	});

	it("should return undefined for an unknown quantization", () => {
		expect(getGGUFQuantBits("NOT_A_QUANT")).toBeUndefined();
	});
});

describe("attentionConfigFromModelConfig", () => {
	it("should read the flat form of a config", () => {
		expect(
			attentionConfigFromModelConfig({
				num_hidden_layers: 32,
				num_attention_heads: 32,
				num_key_value_heads: 8,
				hidden_size: 4096,
				max_position_embeddings: 8192,
			}),
		).toEqual({
			numHiddenLayers: 32,
			numAttentionHeads: 32,
			numKeyValueHeads: 8,
			headDim: undefined,
			hiddenSize: 4096,
			maxPositionEmbeddings: 8192,
			layerTypes: undefined,
		});
	});

	it("should read the nested form of a multimodal config", () => {
		expect(
			attentionConfigFromModelConfig({
				model_type: "qwen3_5",
				text_config: {
					num_hidden_layers: 64,
					num_attention_heads: 24,
					num_key_value_heads: 4,
					head_dim: 256,
					max_position_embeddings: 262144,
				},
			}),
		).toMatchObject({
			numHiddenLayers: 64,
			numAttentionHeads: 24,
			numKeyValueHeads: 4,
			headDim: 256,
			maxPositionEmbeddings: 262144,
		});
	});

	it("should return undefined when the config has no attention", () => {
		expect(attentionConfigFromModelConfig({ model_type: "bert" })).toBeUndefined();
		expect(attentionConfigFromModelConfig(undefined)).toBeUndefined();
	});
});

describe("kvCacheBytesPerToken", () => {
	it("should count the full attention layers only", () => {
		// 64 layers, one full attention layer out of four: 16 layers of KV cache
		// 2 (K and V) * 16 layers * 4 kv heads * 256 head dim * 2 bytes
		expect(
			kvCacheBytesPerToken({
				numHiddenLayers: 64,
				numAttentionHeads: 24,
				numKeyValueHeads: 4,
				headDim: 256,
				layerTypes: Array.from({ length: 64 }, (_, index) => (index % 4 === 3 ? "full_attention" : "linear_attention")),
			}),
		).toBe(65536);
	});

	it("should count every layer of a plain model", () => {
		// 2 * 32 layers * 8 kv heads * 128 head dim * 2 bytes
		expect(
			kvCacheBytesPerToken({
				numHiddenLayers: 32,
				numAttentionHeads: 32,
				numKeyValueHeads: 8,
				headDim: 128,
			}),
		).toBe(131072);
	});
});

describe("getKvCacheSpec", () => {
	it("should describe the kv cache of a model with full attention layers", () => {
		const spec = getKvCacheSpec({
			numHiddenLayers: 32,
			numAttentionHeads: 32,
			numKeyValueHeads: 8,
			headDim: 128,
		});
		if (!spec) {
			throw new Error("the model should have a kv cache");
		}

		expect(spec.bytesPerToken).toBe(131072);
		expect(spec.slidingWindowBytes).toBeUndefined();
		expect(getKvCacheBytes(spec, 4096)).toBe(131072 * 4096);
	});

	it("should keep the sliding attention layers in a bounded cache", () => {
		const spec = getKvCacheSpec({
			numHiddenLayers: 32,
			numAttentionHeads: 32,
			numKeyValueHeads: 8,
			headDim: 128,
			slidingWindow: 4096,
			layerTypes: Array.from({ length: 32 }, (_, index) => (index % 4 === 3 ? "full_attention" : "sliding_attention")),
		});
		if (!spec) {
			throw new Error("the model should have a kv cache");
		}

		// 8 full attention layers
		expect(spec.bytesPerToken).toBe(2 * 8 * 8 * 128 * 2);
		// 24 sliding attention layers, that only hold the last 4096 tokens
		expect(spec.slidingWindow).toBe(4096);
		expect(spec.slidingWindowBytes).toBe(2 * 24 * 8 * 128 * 2 * 4096);

		// The cache doesn't grow with the context past the sliding window
		expect(getKvCacheBytes(spec, 8192)).toBe(spec.bytesPerToken * 8192 + (spec.slidingWindowBytes ?? 0));
	});

	it("should treat the sliding layers as full attention when the window is disabled", () => {
		const spec = getKvCacheSpec({
			numHiddenLayers: 32,
			numAttentionHeads: 32,
			numKeyValueHeads: 8,
			headDim: 128,
			slidingWindow: 4096,
			useSlidingWindow: false,
			layerTypes: Array.from({ length: 32 }, (_, index) => (index % 4 === 3 ? "full_attention" : "sliding_attention")),
		});

		expect(spec?.bytesPerToken).toBe(2 * 32 * 8 * 128 * 2);
		expect(spec?.slidingWindowBytes).toBeUndefined();
	});

	it("should not count the linear attention layers", () => {
		const spec = getKvCacheSpec({
			numHiddenLayers: 64,
			numAttentionHeads: 24,
			numKeyValueHeads: 4,
			headDim: 256,
			layerTypes: Array.from({ length: 64 }, (_, index) => (index % 4 === 3 ? "full_attention" : "linear_attention")),
		});

		expect(spec?.bytesPerToken).toBe(65536);
	});
});

describe("estimateMemoryRequirement", () => {
	it("should add the kv cache and the overhead to the weights", () => {
		const requirement = estimateMemoryRequirement({
			weightBytes: 30_879_676_248,
			kvCache: { bytesPerToken: 65536 },
			contextLength: 8192,
			overheadRatio: 0.05,
			minOverheadBytes: 0,
		});

		expect(requirement.weightBytes).toBe(30_879_676_248);
		expect(requirement.kvCacheBytes).toBe(65536 * 8192);
		expect(requirement.overheadBytes).toBe(Math.round(30_879_676_248 * 0.05));
		expect(requirement.totalBytes).toBe(requirement.weightBytes + requirement.kvCacheBytes + requirement.overheadBytes);
	});
});

describe("computeFit", () => {
	const requirement = { weightBytes: 10, kvCacheBytes: 0, overheadBytes: 0, totalBytes: 10 };

	it("should tell when a model fits", () => {
		expect(computeFit(requirement, 100).state).toBe("fits");
	});

	it("should tell when a model fits tightly", () => {
		expect(computeFit(requirement, 11).state).toBe("tight");
	});

	it("should tell when a model does not fit", () => {
		expect(computeFit(requirement, 9).state).toBe("no-fit");
		expect(computeFit(requirement, 0).state).toBe("no-fit");
	});

	it("should tell that a model needs a shorter context", () => {
		const withContext = { weightBytes: 10, kvCacheBytes: 40, overheadBytes: 0, totalBytes: 50 };

		const fit = computeFit(withContext, 20);

		// The weights fit, but not with the KV cache of the whole context
		expect(fit.state).toBe("tight");
		expect(fit.weightsFit).toBe(true);
	});

	it("should tell that the weights of a model do not fit", () => {
		const withContext = { weightBytes: 10, kvCacheBytes: 40, overheadBytes: 0, totalBytes: 50 };

		const fit = computeFit(withContext, 5);

		expect(fit.state).toBe("no-fit");
		expect(fit.weightsFit).toBe(false);
	});
});

describe("getQuantVariantMemoryRequirement", () => {
	it("should use the context length of the row by default", () => {
		const requirement = getQuantVariantMemoryRequirement({
			id: "Qwen/Qwen3.8-27B-FP8",
			label: "FP8",
			format: "fp8",
			bits: 8,
			sizeBytes: 30_879_676_248,
			contextLength: 4096,
			kvCache: { bytesPerToken: 65536 },
		});

		expect(requirement.kvCacheBytes).toBe(65536 * 4096);
	});

	it("should use the requested context length", () => {
		const requirement = getQuantVariantMemoryRequirement(
			{
				id: "Qwen/Qwen3.8-27B-FP8",
				label: "FP8",
				format: "fp8",
				bits: 8,
				sizeBytes: 1000,
				contextLength: 4096,
				kvCache: { bytesPerToken: 100 },
			},
			{ contextLength: 128 },
		);

		expect(requirement.kvCacheBytes).toBe(12800);
	});
});

describe("formatBytes", () => {
	it("should format a size the way model pages do", () => {
		expect(formatBytes(16056478688)).toBe("16.1 GB");
		expect(formatBytes(30_879_676_248)).toBe("30.9 GB");
		expect(formatBytes(900)).toBe("900 B");
	});
});
