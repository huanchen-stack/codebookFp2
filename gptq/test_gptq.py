import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from fakequant import CodebookQuantizer
from gptq import CodebookGPTQ

torch.manual_seed(42)

OUT, IN = 256, 512
NUM_SAMPLES = 1024

W = torch.randn(OUT, IN) * 0.3
X = torch.randn(NUM_SAMPLES, IN) * 0.5

q = CodebookQuantizer()

print(f"=== GPTQ test: linear layer ({OUT}x{IN}), {NUM_SAMPLES} calibration samples ===")
print(f"  W range: [{W.min().item():.4f}, {W.max().item():.4f}]")
print(f"  X range: [{X.min().item():.4f}, {X.max().item():.4f}]")

Y_ref = X @ W.T

print("\n--- Plain codebook (no GPTQ) ---")
blocks_plain = W.reshape(-1, 16)
fp4_plain, scale_plain = q.fakequant_blocks_with_scale(blocks_plain)
W_q_plain = (fp4_plain * scale_plain).reshape(OUT, IN)
Y_plain = X @ W_q_plain.T

raw_mse_plain = ((W - W_q_plain) ** 2).mean().item()
output_mse_plain = ((Y_ref - Y_plain) ** 2).mean().item()
print(f"  weight MSE:  {raw_mse_plain:.6f}")
print(f"  output MSE:  {output_mse_plain:.6f}")

print("\n--- GPTQ codebook (with importance weighting) ---")
gptq = CodebookGPTQ(in_features=IN, quantizer=q, use_importance=True)
gptq.update(X)
fp4_gptq, scale_gptq, W_q_gptq, fp4_phase1 = gptq.quantize(W)
Y_gptq = X @ W_q_gptq.T

raw_mse_gptq = ((W - W_q_gptq) ** 2).mean().item()
output_mse_gptq = ((Y_ref - Y_gptq) ** 2).mean().item()
print(f"  weight MSE:  {raw_mse_gptq:.6f}")
print(f"  output MSE:  {output_mse_gptq:.6f}")

print("\n--- Comparison ---")
weight_ratio = raw_mse_gptq / raw_mse_plain
output_ratio = output_mse_gptq / output_mse_plain
print(f"  weight MSE ratio (GPTQ/plain): {weight_ratio:.4f}")
print(f"  output MSE ratio (GPTQ/plain): {output_ratio:.4f}")
print(f"  output MSE reduction:          {(1 - output_ratio) * 100:.1f}%")

blocks_gptq = fp4_gptq.reshape(-1, 16)
max_distinct = max(len(set(blocks_gptq[i].tolist())) for i in range(blocks_gptq.shape[0]))
print(f"  max distinct per block:        {max_distinct}")

assert max_distinct <= 4, f"GPTQ produced block with {max_distinct} distinct values"
assert output_mse_gptq < output_mse_plain, \
    f"GPTQ output MSE ({output_mse_gptq:.6f}) should be < plain ({output_mse_plain:.6f})"
assert output_mse_gptq < output_mse_plain * 0.80, \
    f"GPTQ output MSE reduction ({(1 - output_ratio) * 100:.1f}%) should be >= 20%"
assert not W_q_gptq.isnan().any(), "NaN in GPTQ output"

print("\n--- Intra-group column-by-column changes ---")
blocks_phase1 = fp4_plain.reshape(-1, 16)
blocks_phase2 = fp4_gptq.reshape(-1, 16)
total_elements = blocks_phase2.numel()
changed_elements = (blocks_phase1 != blocks_phase2).sum().item()
changed_blocks = ((blocks_phase1 != blocks_phase2).any(dim=1)).sum().item()
total_blocks = blocks_phase2.shape[0]
print(f"  all columns:")
print(f"    elements changed:  {changed_elements}/{total_elements} ({100 * changed_elements / total_elements:.1f}%)")
print(f"    blocks changed:    {changed_blocks}/{total_blocks} ({100 * changed_blocks / total_blocks:.1f}%)")

first16_plain = fp4_plain.reshape(OUT, IN)[:, :16]
first16_gptq = fp4_gptq[:, :16]
first16_total = first16_gptq.numel()
first16_changed = (first16_plain != first16_gptq).sum().item()
first16_per_col = [(first16_plain[:, j] != first16_gptq[:, j]).sum().item() for j in range(16)]
print(f"  first 16 columns (no inter-block compensation):")
print(f"    elements changed:  {first16_changed}/{first16_total} ({100 * first16_changed / first16_total:.1f}%)")
print(f"    per column:        {first16_per_col}")

print("\n--- Phase 1 vs Phase 2 (intra-group column-wise effect) ---")
total_el = fp4_gptq.numel()
changed_el = (fp4_phase1 != fp4_gptq).sum().item()
changed_per_group: list[int] = []
for g_start in range(0, IN, 16):
    g_end = min(g_start + 16, IN)
    diff = (fp4_phase1[:, g_start:g_end] != fp4_gptq[:, g_start:g_end]).sum().item()
    changed_per_group.append(diff)
group_size = OUT * 16
print(f"  total elements flipped:  {changed_el}/{total_el} ({100 * changed_el / total_el:.1f}%)")
print(f"  per 16-col group (first 8):  {[f'{c}/{group_size} ({100*c/group_size:.1f}%)' for c in changed_per_group[:8]]}")
print(f"  per 16-col group (last 8):   {[f'{c}/{group_size} ({100*c/group_size:.1f}%)' for c in changed_per_group[-8:]]}")
avg_changed = sum(changed_per_group) / len(changed_per_group)
print(f"  avg per group:               {avg_changed:.1f}/{group_size} ({100 * avg_changed / group_size:.1f}%)")

print("\n--- Codebook consistency ---")
cb = q.codebook
codebook_ok = True
for i in range(blocks_gptq.shape[0]):
    block_vals = set(blocks_gptq[i].tolist())
    found = False
    for j in range(cb.shape[0]):
        if block_vals <= set(cb[j].tolist()):
            found = True
            break
    if not found:
        print(f"  FAIL: block {i} values {block_vals} not in any codebook entry")
        codebook_ok = False
        break
if codebook_ok:
    print(f"  All {blocks_gptq.shape[0]} blocks use valid codebook entries")
assert codebook_ok, "GPTQ output contains blocks not matching any codebook entry"

print("\n--- GPTQ codebook (without importance weighting) ---")
gptq_noimp = CodebookGPTQ(in_features=IN, quantizer=q, use_importance=False)
gptq_noimp.update(X)
_, _, W_q_noimp, _ = gptq_noimp.quantize(W)
Y_noimp = X @ W_q_noimp.T
output_mse_noimp = ((Y_ref - Y_noimp) ** 2).mean().item()
raw_mse_noimp = ((W - W_q_noimp) ** 2).mean().item()
print(f"  weight MSE:  {raw_mse_noimp:.6f}")
print(f"  output MSE:  {output_mse_noimp:.6f}")

print("\n--- Importance weighting ablation ---")
print(f"  {'':30s} {'no importance':>14s} {'with importance':>16s}")
print(f"  {'weight MSE':30s} {raw_mse_noimp:14.6f} {raw_mse_gptq:16.6f}")
print(f"  {'output MSE':30s} {output_mse_noimp:14.6f} {output_mse_gptq:16.6f}")
print(f"  {'output MSE ratio (vs plain)':30s} {output_mse_noimp / output_mse_plain:14.4f} {output_mse_gptq / output_mse_plain:16.4f}")
print(f"  {'output MSE reduction':30s} {(1 - output_mse_noimp / output_mse_plain) * 100:13.1f}% {(1 - output_mse_gptq / output_mse_plain) * 100:15.1f}%")
imp_improvement = (1 - output_mse_gptq / output_mse_noimp) * 100
print(f"  importance weighting effect:  {imp_improvement:+.2f}%")

print("\n--- Importance weight distribution (Gaussian activations) ---")
H_gauss = X.T @ X / NUM_SAMPLES
h_diag_gauss = H_gauss.diag()
print(f"  H.diag() range:  [{h_diag_gauss.min().item():.4f}, {h_diag_gauss.max().item():.4f}]")
print(f"  H.diag() mean:   {h_diag_gauss.mean().item():.4f}")
print(f"  max/min ratio:   {h_diag_gauss.max().item() / h_diag_gauss.min().item():.2f}x")
h_sorted = h_diag_gauss.sort(descending=True).values
print(f"  top-5:    {h_sorted[:5].tolist()}")
print(f"  bottom-5: {h_sorted[-5:].tolist()}")

print("\n--- Hessian-weighted MSE (Gaussian) ---")
delta_plain = W - W_q_plain
delta_gptq = W - W_q_gptq
delta_noimp = W - W_q_noimp
h_mse_plain = (delta_plain @ H_gauss @ delta_plain.T).trace().item() / OUT
h_mse_gptq = (delta_gptq @ H_gauss @ delta_gptq.T).trace().item() / OUT
h_mse_noimp = (delta_noimp @ H_gauss @ delta_noimp.T).trace().item() / OUT
print(f"  plain:            {h_mse_plain:.6f}")
print(f"  GPTQ (no imp):    {h_mse_noimp:.6f}  (ratio: {h_mse_noimp / h_mse_plain:.4f})")
print(f"  GPTQ (with imp):  {h_mse_gptq:.6f}  (ratio: {h_mse_gptq / h_mse_plain:.4f})")

assert h_mse_gptq < h_mse_plain, \
    f"GPTQ Hessian MSE ({h_mse_gptq:.6f}) should be < plain ({h_mse_plain:.6f})"

print("\n" + "=" * 70)
print("=== Outlier activation test ===")
print("=" * 70)

torch.manual_seed(99)
OUT2, IN2 = 256, 512
NUM_SAMPLES2 = 1024

W2 = torch.randn(OUT2, IN2) * 0.3
X2 = torch.randn(NUM_SAMPLES2, IN2) * 0.5

outlier_channels = [3, 7, 19, 35, 48, 67, 99, 128, 200, 300, 400, 500]
for ch in outlier_channels:
    if ch < IN2:
        X2[:, ch] *= 50.0

Y2_ref = X2 @ W2.T

H2 = X2.T @ X2 / NUM_SAMPLES2
h_diag2 = H2.diag()
print(f"\n--- Importance weight distribution (outlier activations) ---")
print(f"  H.diag() range:  [{h_diag2.min().item():.4f}, {h_diag2.max().item():.4f}]")
print(f"  H.diag() mean:   {h_diag2.mean().item():.4f}")
print(f"  max/min ratio:   {h_diag2.max().item() / h_diag2.min().item():.1f}x")
h_sorted2 = h_diag2.sort(descending=True).values
print(f"  top-5:    {[f'{v:.2f}' for v in h_sorted2[:5].tolist()]}")
print(f"  bottom-5: {[f'{v:.4f}' for v in h_sorted2[-5:].tolist()]}")

imp_per_group: list[float] = []
for g_start in range(0, IN2, 16):
    g_end = min(g_start + 16, IN2)
    group_imp = h_diag2[g_start:g_end]
    normalized = group_imp / group_imp.mean().clamp(min=1e-10)
    imp_per_group.append(normalized.max().item())
print(f"  max normalized importance within group (first 8): {[f'{v:.1f}x' for v in imp_per_group[:8]]}")
print(f"  groups with >5x outlier: {sum(1 for v in imp_per_group if v > 5)}/{len(imp_per_group)}")

print("\n--- Plain codebook (outlier) ---")
blocks_plain2 = W2.reshape(-1, 16)
fp4_plain2, scale_plain2 = q.fakequant_blocks_with_scale(blocks_plain2)
W_q_plain2 = (fp4_plain2 * scale_plain2).reshape(OUT2, IN2)
Y_plain2 = X2 @ W_q_plain2.T
output_mse_plain2 = ((Y2_ref - Y_plain2) ** 2).mean().item()
print(f"  output MSE:  {output_mse_plain2:.6f}")

print("\n--- GPTQ without importance (outlier) ---")
gptq2_noimp = CodebookGPTQ(in_features=IN2, quantizer=q, use_importance=False)
gptq2_noimp.update(X2)
_, _, W_q2_noimp, _ = gptq2_noimp.quantize(W2)
Y2_noimp = X2 @ W_q2_noimp.T
output_mse2_noimp = ((Y2_ref - Y2_noimp) ** 2).mean().item()
print(f"  output MSE:  {output_mse2_noimp:.6f}")
print(f"  reduction vs plain: {(1 - output_mse2_noimp / output_mse_plain2) * 100:.1f}%")

print("\n--- GPTQ with importance (outlier) ---")
gptq2_imp = CodebookGPTQ(in_features=IN2, quantizer=q, use_importance=True)
gptq2_imp.update(X2)
_, _, W_q2_imp, _ = gptq2_imp.quantize(W2)
Y2_imp = X2 @ W_q2_imp.T
output_mse2_imp = ((Y2_ref - Y2_imp) ** 2).mean().item()
print(f"  output MSE:  {output_mse2_imp:.6f}")
print(f"  reduction vs plain: {(1 - output_mse2_imp / output_mse_plain2) * 100:.1f}%")

print(f"\n--- Importance weighting ablation (outlier) ---")
print(f"  {'':30s} {'no importance':>14s} {'with importance':>16s}")
print(f"  {'output MSE':30s} {output_mse2_noimp:14.6f} {output_mse2_imp:16.6f}")
print(f"  {'reduction vs plain':30s} {(1 - output_mse2_noimp / output_mse_plain2) * 100:13.1f}% {(1 - output_mse2_imp / output_mse_plain2) * 100:15.1f}%")
imp_improvement2 = (1 - output_mse2_imp / output_mse2_noimp) * 100
print(f"  importance weighting effect:  {imp_improvement2:+.2f}%")

h_mse2_plain = (((W2 - W_q_plain2) @ H2) * (W2 - W_q_plain2)).sum().item() / OUT2
h_mse2_noimp = (((W2 - W_q2_noimp) @ H2) * (W2 - W_q2_noimp)).sum().item() / OUT2
h_mse2_imp = (((W2 - W_q2_imp) @ H2) * (W2 - W_q2_imp)).sum().item() / OUT2
print(f"\n--- Hessian-weighted MSE (outlier) ---")
print(f"  plain:            {h_mse2_plain:.6f}")
print(f"  GPTQ (no imp):    {h_mse2_noimp:.6f}  (ratio: {h_mse2_noimp / h_mse2_plain:.4f})")
print(f"  GPTQ (with imp):  {h_mse2_imp:.6f}  (ratio: {h_mse2_imp / h_mse2_plain:.4f})")
h_imp_effect = (1 - h_mse2_imp / h_mse2_noimp) * 100
print(f"  importance weighting effect on H-MSE: {h_imp_effect:+.2f}%")

print("\nAll GPTQ tests passed.")
