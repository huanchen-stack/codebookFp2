# Codebook Quantization: NVFP4 → INT2

## Core Idea

Use a codebook to further quantize NVFP4 weights to ~2 bits per element.

## NVFP4 Baseline

- Each block: 16 elements, 4 bits each (E2M1 format)
- 15 representable values (including zero):
  - Magnitudes: {0, 0.5, 1, 1.5, 2, 3, 4, 6}
  - With sign: 8 positive + 7 negative (zero has no sign) = 15 values
  - So 14 non-zero values total
- Per-block scaling: FP8 scale (per group of 16) + FP32 global scale (per tensor)
- Dequant formula: `W = fp4_value * (weight_scale / weight_global_scale)`

## Codebook Scheme (Primary)

Pick 3 non-zero values out of the 14 non-zero FP4 values → each element maps to one of 4 options (3 chosen + zero) → 2 bits per element.

- Codebook size: C(14, 3) = 364 entries
- For fake quantization: use the full 364-entry codebook (no sign optimization)
  - Sign trick (halving to 182 by folding sign into scale factor) is deferred — too complex for initial implementation
- Codebook index: ceil(log2(364)) = 9 bits per block

### Bit Budget per Block

| Component          | NVFP4 (original) | Codebook INT2    |
|--------------------|-------------------|------------------|
| Element values     | 16 × 4 = 64 bits | 16 × 2 = 32 bits |
| Codebook index     | —                 | 9 bits            |
| Scale factor       | 8 bits (FP8)     | 8 bits (FP8)     |
| **Total per block** | **72 bits**       | **49 bits**       |

~32% reduction in storage.

## Codebook Selection (Quantization)

For each 16-element block:
1. Try all 364 codebook entries
2. For each entry: map every element to the nearest of the 4 allowed values (3 chosen + zero)
3. Compute per-block MSE against the original FP4 values
4. Pick the codebook entry with minimum MSE

## Dequantization

Read the codebook index → recover which 3 values were used → reconstruct the 4-valued FP4 block.

## Extensibility

The codebook class should support alternative policies (not implemented now):
- Pick only non-zero values (no zero in the 4 options)
- Pair positive/negative of the same magnitude together
- Sign-folding optimization (182 entries + scale sign flip)
- Other strategies
