# GPTQ-Calibrated Codebook Selection

## The Problem

Current codebook selection in `_fakequant_blocks_chunk()` picks the codebook with minimum **naive MSE** over raw FP4 values. This treats every weight equally — a weight that barely matters for the output is penalized the same as a weight that dominates the output. GPTQ fixes this by weighting the error by each weight's actual importance (measured from calibration data).

## Why This Is Hard (and the Plan to Make It Not)

Two challenges:
1. **Scaling factors**: MSE is currently on raw FP4. GPTQ needs to account for `weight_scale / weight_global_scale`.
2. **Forward pass**: Computing importance requires running calibration data through the model.

Good news on (1): within each block of 16 elements, the scale factor is **constant** (one FP8 scale per group of 16). So for codebook selection *within* a block, the scale cancels out — we only need to weight by the Hessian diagonal. The scale matters for error propagation *across* blocks (Phase 2).

## Reference Implementation: FP-Quant

`/home/huanchen/FP-Quant/src/quantization/gptq.py` already adapts GPTQ for NVFP4. Reuse directly:

| FP-Quant component | File & lines | Reuse? |
|---|---|---|
| `GPTQ.update()` — Hessian accumulation | `gptq.py:68-102` | **Copy verbatim** |
| `GPTQ._get_hessian_inverse()` — damping + Cholesky | `gptq.py:199-218` | **Copy verbatim** |
| `accumulate_hessian.py` — Triton kernel for `H += X^T X` | entire file | **Copy verbatim** (or fallback to `H += X.T @ X` if no Triton) |
| Calibration loop — layer-by-layer, hooks, activation updates | `gptq.py:225-470` | **Adapt for MoE** (hook each expert separately) |
| `GPTQ.step()` inner loop — per-column quantize | `gptq.py:159-187` | **Modify**: step by 16 columns, replace scalar FP4 round with codebook selection |
| `Quantizer` class — FP4 scale/round | `quantizer.py` | **Not needed** (we have our own `CodebookQuantizer`) |

**The only code to write**: modify the inner loop in `step()` to process 16 columns at a time and call our `_fakequant_blocks_chunk()` instead of their scalar `quantizer()`. Everything else (Hessian, Cholesky, error propagation, calibration) is reused.

## Three Phases (Increasing Complexity)

### Phase 1: Hessian-Diagonal Weighted MSE [START HERE]

**What changes**: Replace `mse_all.mean(dim=1)` with `(hessian_diag * element_errors).mean(dim=1)`.

**Effort**: ~5 lines changed in `fakequant.py` + new `calibrate.py` (reuse GPTQ's `add_batch()` Hessian accumulator, extract `.diag()`).

**Expected gain**: Moderate. Important weights get protected; unimportant weights absorb more error. Should close ~30-50% of the quality gap vs lossless.

**How it works**:
```
Currently:   mse_all[c] = mean_over_16_elements( (fp4_orig - fp4_quant)² )
Proposed:    mse_all[c] = mean_over_16_elements( H_diag * (fp4_orig - fp4_quant)² )
```

Where `H_diag[col]` = diagonal of the Hessian = importance of column `col` = `(2/N) * sum(x_col²)` across calibration samples. High `H_diag` means this weight column has high-magnitude activations flowing through it.

**Diff for `_fakequant_blocks_chunk()`**:
```python
# BEFORE (line 183-189):
element_errors = error_table[element_indices]
mse_all = element_errors.mean(dim=1)

# AFTER:
element_errors = error_table[element_indices]        # [num_blocks, 16, 364]
if hessian_diag is not None:
    h = hessian_diag.unsqueeze(-1)                   # [num_blocks, 16, 1]
    mse_all = (h * element_errors).sum(dim=1) / h.sum(dim=1).clamp(min=1e-8)
else:
    mse_all = element_errors.mean(dim=1)             # [num_blocks, 364]
```

**Files**:
- NEW `calibrate.py` — reuse GPTQ's Hessian accumulation pattern (`add_batch()`), save `H.diag()` per layer
- EDIT `fakequant.py` — add `hessian_diag` param (~5 lines)
- EDIT `fakequant_model.py` — load Hessians, pass through

---

### Phase 2: Full GPTQ Error Propagation

**What changes**: After choosing a codebook for block `b`, propagate the quantization error forward to blocks `b+1, b+2, ...` using the off-diagonal Hessian.

**Effort**: Copy FP-Quant's `GPTQ.step()` (lines 159-187 of `/home/huanchen/FP-Quant/src/quantization/gptq.py`). Modify the inner loop from per-column to per-16-column-block.

**Expected gain**: Large. This is the core GPTQ innovation. Should close ~60-80% of the remaining gap after Phase 1.

**Concrete diff to FP-Quant's `step()` inner loop**:
```python
# FP-Quant original (per-column, lines 166-185):
for i in range(ncols):
    w_ci = w_blk[:, i]
    d = H_inv_cho_blk[i, i]
    g_idx = (c1 + i) // group_size
    w_q = self.quantizer(w_ci, scales[:, g_idx], zeros[:, g_idx])
    err = (w_ci - w_q) / d
    w_blk[:, i:].addr_(err, H_inv_cho_blk[i, i:], alpha=-1)
    errs[:, i] = err

# Our version (per-16-column-block):
for i in range(0, ncols, 16):
    w_block = w_blk[:, i:i+16]                          # [out, 16] real-valued
    H_diag_block = H_inv_cho_blk[i:i+16, i:i+16]
    scale = weight_scale[:, (c1+i)//16] / gscale         # per-group scale

    # Convert to FP4 space → codebook select → convert back
    fp4_block = cast_to_fp4(w_block / scale.unsqueeze(1))
    fp4_quant = codebook_quantizer.fakequant_blocks(fp4_block.reshape(-1, 16))
    fp4_quant = fp4_quant.reshape_as(fp4_block)
    w_q = fp4_quant * scale.unsqueeze(1)

    # Error propagation (same math, just 16 columns at once)
    err_block = (w_block - w_q)                          # [out, 16]
    # Simplified: use diagonal of H_inv for the block
    for j in range(16):
        d = H_inv_cho_blk[i+j, i+j]
        err_col = err_block[:, j] / d
        w_blk[:, i+j+1:].addr_(err_col, H_inv_cho_blk[i+j, i+j+1:], alpha=-1)
        errs[:, i+j] = err_col
```

Note: the inner `for j in range(16)` still propagates error column-by-column within the block — we just do codebook selection on all 16 at once. This preserves GPTQ's exact error propagation math.

**Files**:
- EDIT `calibrate.py` — save full Hessian `H` [in_features, in_features]
- NEW `gptq_quantize.py` — FP-Quant's GPTQ class with modified `step()`
- EDIT `fakequant_model.py` — `--mode=gptq` flag

---

### Phase 3: Iterative Refinement

**What changes**: After the initial GPTQ pass, re-visit each block and re-select its codebook given the final state of all other blocks. Repeat for T iterations.

**Effort**: ~100 lines on top of Phase 2.

**Expected gain**: Small (diminishing returns). Closes the last 5-15% of gap.

**Implementation**: Wrap the Phase 2 block loop in an outer iteration. On iteration t > 0, "undo" the current block's codebook (restore to original + accumulated error), then re-select. QuIP# calls this `quip_tune_iters` and typically uses 0-10 iterations.

---

## Calibration Data Details

**What to use**: C4 (following GPTQ original) or RedPajama (following QuIP#/AQLM). Both work fine.

**How much**: 128 sequences × 2048 tokens = ~260K tokens. More helps but diminishing returns past 256 sequences.

**How to collect for MoE experts**: The router decides which tokens go to which expert. We need per-expert Hessians.
1. Run full model forward pass with hooks on EVERY expert's linear layer
2. For expert `e`, only the tokens routed to it contribute to its Hessian
3. Result: `H_expert_e` has lower rank than a dense model's H (fewer tokens), but accurately reflects this expert's actual workload
4. Concern: some experts may see very few tokens → their Hessians may be noisy. Mitigation: use more calibration data (256+ sequences) or add stronger damping for low-sample experts.

**Hook pattern for Hessian accumulation** (GPTQ standard):
```python
def make_hook(layer_name, hessians, sample_counts):
    def hook(module, input, output):
        X = input[0].float().reshape(-1, input[0].shape[-1])  # [tokens, hidden]
        n = X.shape[0]
        hessians[layer_name] *= sample_counts[layer_name] / (sample_counts[layer_name] + n)
        sample_counts[layer_name] += n
        X = (2 / sample_counts[layer_name]) ** 0.5 * X
        hessians[layer_name] += X.T @ X
    return hook
```

**Memory for Qwen3-30B-A3B Hessians**:
- Dense layers: `in_features` up to 3584 → H is 3584² × 4 bytes ≈ 49 MB per layer
- Expert layers: smaller `in_features` (MoE experts are narrower) → even cheaper
- Diagonal only (Phase 1): [in_features] × 4 bytes ≈ 14 KB per layer — trivial
- Full H on disk for all layers: ~200 layers × 50 MB ≈ 10 GB total. Fine.

**Loading the model for calibration**:
- Need the model in transformers format (not TRT) to register forward hooks
- Option A: Load with `AutoModelForCausalLM` + `nvidia-modelopt` to restore NVFP4 state
- Option B: Manually dequantize — load uint8 packed + scales, convert to BF16, build vanilla model
- Option A is cleaner if modelopt hook access works; try that first

---

## Execution Order

1. **Implement Phase 1** — highest ROI: simple change, meaningful quality gain
2. **Benchmark Phase 1** on full eval suite (GSM8K, GPQA Diamond, MMLU, MMMU)
3. If gap is <0.5pt on GSM8K → Phase 1 may suffice for the paper
4. If gap is >0.5pt → implement Phase 2, which should close most remaining gap
5. Phase 3 only if Phase 2 still leaves a noticeable gap

## File Changes Summary

```
Phase 1:
  NEW   calibrate.py              # Reuse GPTQ's add_batch() pattern, save H.diag()
  EDIT  fakequant.py              # ~5 lines: hessian_diag param in codebook selection
  EDIT  fakequant_model.py        # Load Hessians, pass through

Phase 2:
  EDIT  calibrate.py              # Save full H (not just diagonal)
  NEW   gptq_quantize.py          # GPTQ's fasterquant() with codebook_select() swapped in
  EDIT  fakequant_model.py        # --mode=gptq flag

Phase 3:
  EDIT  gptq_quantize.py          # Wrap block loop in outer iteration
```

## Reference Weight Mode: BF16 vs NVFP4

The GPTQ objective is `minimize ||X(W_ref - W_cbint2)||²`. The choice of `W_ref` matters:

| Mode | Reference | What it minimizes | When to use |
|---|---|---|---|
| `--ref=fp4` | NVFP4 weights | Codebook deviation from FP4 | Default. Preserves existing FP4 quality. Hessian computed from FP4 model forward pass. |
| `--ref=bf16` | Original BF16 weights | Codebook deviation from full precision | Ambitious. May recover quality *beyond* NVFP4 baseline if codebook finds a better local optimum. |

**Implementation**:
- `--ref=fp4`: Load NVFP4 model, dequantize FP4 weights as `W_ref = fp4 * (scale / gscale)`. Hessian from FP4 model activations.
- `--ref=bf16`: Load original BF16 model (e.g., `Qwen/Qwen3-30B-A3B`). Use BF16 weights as `W_ref`. Hessian from BF16 model activations.

For `--ref=bf16`, the GPTQ loop works in BF16 real space. At each block:
1. Take corrected BF16 weights (with error propagation from previous blocks)
2. Divide by NVFP4 scale → snap to FP4 grid → codebook select → multiply by scale
3. Error = `W_bf16_corrected - W_cbint2_real` (real space)
4. Propagate error forward

This is strictly more powerful than `--ref=fp4` because it optimizes against the true target, but it requires access to the original BF16 model weights.

**Recommendation**: Implement `--ref=fp4` first (simpler, only needs the NVFP4 model). Add `--ref=bf16` as a second option. Compare both in the paper — if `--ref=bf16` recovers quality beyond the NVFP4 baseline, that's an extra result worth showing.

## Open Questions

- [ ] Which calibration dataset? C4 is easiest (`datasets` library), RedPajama closer to pretraining dist
- [ ] How to load the model for calibration? Test if `nvidia-modelopt` supports forward hooks
- [ ] Per-expert Hessians: do some experts see too few tokens? May need more calibration data or stronger damping
- [ ] FP32 vs FP64 Hessian accumulation: try FP32 first, switch to FP64 if numerical issues
- [ ] For `--ref=bf16`: need access to `Qwen/Qwen3-30B-A3B` base model weights. Confirm availability on HF.
