# Implementation Plan

## Source Model

- **Model**: `nvidia/Qwen3-30B-A3B-NVFP4` (MoE, 30B total / 3B active)
- **Format**: `modelopt` quantization method (NOT `compressed-tensors`)
- **HF repo**: https://huggingface.co/nvidia/Qwen3-30B-A3B-NVFP4
- **Hardware target**: RTX 6000 Pro (Blackwell — native FP4 tensor core support)

## NVFP4 Weight Format on Disk

Each quantized linear layer has 3 tensors in safetensors:
```
*.weight_packed       → uint8 [out_features, in_features // 2]   # two FP4 nibbles per byte
*.weight_scale        → float8_e4m3fn [out_features, in_features // 16]  # per-group-of-16 scale
*.weight_global_scale → float32 [1]                               # per-tensor global scale
```

Non-quantized layers (BF16): `lm_head`, `model.embed_tokens`, all `model.layers.*.mlp.gate` (MoE routers), layer norms.

Safetensors split into 4 shards (~18.1 GB total), with `model.safetensors.index.json` mapping tensors to shards.

## File Structure

```
.
├── .agent/
│   ├── idea.md          # Codebook quantization concept
│   └── plan.md          # This file
├── downloader.py        # Download nvidia/Qwen3-30B-A3B-NVFP4
├── model.py             # Inference (load + generate)
├── fakequant.py         # Codebook fake-quantization logic
├── fakequant_model.py   # Apply fakequant to full model, save to new path
└── lm_eval.py           # Benchmark on downstream tasks
```

## File Details

### downloader.py
- Download `nvidia/Qwen3-30B-A3B-NVFP4` from HuggingFace
- Use `huggingface_hub.snapshot_download()` or equivalent
- Save to a configurable local path

### model.py
- Load the NVFP4 model for inference
- Approach: use `nvidia-modelopt` to restore quantized state onto the base Qwen3-30B-A3B architecture
  - Alternative: load raw safetensors + manual dequant if modelopt is problematic
- Simplest possible interface: load model, tokenize, generate, decode
- Must work identically whether pointed at the original or fakequant'd model (transparency requirement)

### fakequant.py
- Core codebook logic operating on raw NVFP4 weight tensors
- **Input**: `weight_packed` (uint8), `weight_scale` (FP8), `weight_global_scale` (FP32) for one layer
- **Process per 16-element block**:
  1. Unpack uint8 → 16 FP4 values (using nibble extraction)
  2. Try all 364 codebook entries (C(14,3) combinations of non-zero FP4 values)
  3. For each codebook: map each element to nearest of {zero, val1, val2, val3}
  4. Compute MSE against original FP4 values
  5. Select codebook with minimum MSE
  6. Replace elements with the 4-valued quantized version
- **Output**: modified `weight_packed` (uint8) — same format, only element values changed to ≤4 distinct values per block
- Scale tensors (`weight_scale`, `weight_global_scale`) remain unchanged
- Class design should be extensible for alternative codebook policies

### fakequant_model.py
- Iterate over all quantized layers in the model (all linear layers except lm_head and MoE gates)
- Apply `fakequant.py` to each layer's weight tensors
- Save the result as a new set of safetensors files in the same format
- Must preserve: `model.safetensors.index.json`, `config.json`, `tokenizer*`, all non-quantized tensors
- Output directory is a drop-in replacement for the original model path

### lm_eval.py
- Benchmark using `lm-evaluation-harness` or manual evaluation
- Target tasks: **GSM8k, HellaSwag, WinoGrande**
- Compare: original NVFP4 model vs. fakequant'd model
- Report accuracy/score for each task

## Dependencies

```
torch
transformers
safetensors
huggingface_hub
nvidia-modelopt[torch]   # for model loading/inference
lm-eval                  # for benchmarking (if using harness)
```

## Key Technical Notes

1. **Packing**: Two FP4 nibbles per uint8 byte. bits[3:0] = first value, bits[7:4] = second value. Sign = bit 3, magnitude index = bits[2:0].
2. **FP4 E2M1 lookup**: `[0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]` — 8 magnitudes, with sign bit giving 15 unique values.
3. **Dequant**: `W_fp16 = fp4_value * (weight_scale.float() / weight_global_scale)`
4. **Transparency**: fakequant output is byte-identical format to input — only the packed nibble values change. No new tensor names, no format changes.
5. **Loss metric**: Per-block MSE of FP4 values (before scaling) when selecting codebook entries.
