from __future__ import annotations

import json
import tempfile
from pathlib import Path

import torch
from safetensors.torch import save_file

from codebook_analysis import _evaluate_codebooks_batch, run_analysis
from fakequant import CodebookQuantizer

torch.manual_seed(42)

OUT, IN = 128, 512
BATCH = 64
NUM_CODEBOOKS = 256

W = torch.randn(OUT, IN, dtype=torch.bfloat16) 
X = torch.randn(BATCH, IN, dtype=torch.bfloat16)

Y_ref = X.float() @ W.float().T

tmpdir = tempfile.mkdtemp()
model_dir = Path(tmpdir) / "model"
hessian_dir = Path(tmpdir) / "hessians"
output_dir = Path(tmpdir) / "codebooks"
model_dir.mkdir()
hessian_dir.mkdir()

layer_name = "model.layers.0.mlp.gate_proj"
shard_file = "block_0000.safetensors"
save_file({f"{layer_name}.weight": W}, str(model_dir / shard_file))
weight_map = {f"{layer_name}.weight": shard_file}

index = {"metadata": {}, "weight_map": weight_map}
with (model_dir / "model.safetensors.index.json").open("w") as f:
    json.dump(index, f)

H = (X.float().T @ X.float()) / BATCH
save_file({layer_name: H}, str(hessian_dir / "block_00.safetensors"))

print("=== Running codebook analysis ===")
run_analysis(
    model_path=model_dir,
    hessian_dir=hessian_dir,
    output_dir=output_dir,
    mlp_only=True,
    num_codebooks=NUM_CODEBOOKS,
    selection_method="frequency",
    coverage_threshold=1.05,
    device_str="cpu",
    chunk_size=512,
)

stat_cb_path = output_dir / "model.layers.0.mlp.gate_proj.pt"
stat_cb = torch.load(str(stat_cb_path), weights_only=True)

W_f32 = W.float()
blocks = W_f32.reshape(-1, 16)
h_diag = H.diag()
importance = h_diag.reshape(IN // 16, 16).repeat(OUT, 1)

default_cb = CodebookQuantizer(policy="top3_nonzero").codebook

print(f"\n=== Codebook sets ===")
print(f"  Default (top3_nonzero): {default_cb.shape[0]} entries, each = [0.0, a, b, c]")
print(
    f"  Statistical:            {stat_cb.shape[0]} entries, data-driven from 1365 candidates"
)

print(f"\n{'=' * 70}")
print(f"=== A: Same evaluation method (_evaluate_codebooks_batch) ===")
print(f"{'=' * 70}")

_, mse_default_all = _evaluate_codebooks_batch(blocks, importance, default_cb)
best_mse_default = mse_default_all.min(dim=-1).values
valid_default = torch.isfinite(best_mse_default)

_, mse_stat_all = _evaluate_codebooks_batch(blocks, importance, stat_cb)
best_mse_stat = mse_stat_all.min(dim=-1).values
valid_stat = torch.isfinite(best_mse_stat)

valid_both = valid_default & valid_stat
avg_mse_default_eval = best_mse_default[valid_both].mean().item()
avg_mse_stat_eval = best_mse_stat[valid_both].mean().item()

print(f"\n  Per-block MSE (analysis evaluation, Hessian-weighted):")
print(f"    Default ({default_cb.shape[0]}): {avg_mse_default_eval:.6f}")
print(f"    Statistical ({stat_cb.shape[0]}): {avg_mse_stat_eval:.6f}")
ratio_eval = avg_mse_stat_eval / avg_mse_default_eval
print(f"    Ratio: {ratio_eval:.4f}x  ({'better' if ratio_eval < 1 else 'worse'})")

print(f"\n{'=' * 70}")
print(f"=== B: Same evaluation method (fakequant_blocks_with_scale) ===")
print(f"{'=' * 70}")

q_default = CodebookQuantizer(policy="top3_nonzero")
q_stat = CodebookQuantizer(policy="top3_nonzero")
q_stat.set_codebook(stat_cb)

fp4_default, s_default = q_default.fakequant_blocks_with_scale(
    blocks, importance_weights=importance
)
W_default = (fp4_default * s_default).reshape(OUT, IN)
fp4_stat, s_stat = q_stat.fakequant_blocks_with_scale(
    blocks, importance_weights=importance
)
W_stat = (fp4_stat * s_stat).reshape(OUT, IN)

weight_mse_default = ((W_f32 - W_default) ** 2).mean().item()
weight_mse_stat = ((W_f32 - W_stat) ** 2).mean().item()

Y_default = X.float() @ W_default.T
Y_stat = X.float() @ W_stat.T
output_mse_default = ((Y_ref - Y_default) ** 2).mean().item()
output_mse_stat = ((Y_ref - Y_stat) ** 2).mean().item()

print(f"\n  Weight MSE (fakequant path):")
print(f"    Default ({default_cb.shape[0]}): {weight_mse_default:.6f}")
print(f"    Statistical ({stat_cb.shape[0]}): {weight_mse_stat:.6f}")
ratio_w = weight_mse_stat / weight_mse_default
print(f"    Ratio: {ratio_w:.4f}x  ({'better' if ratio_w < 1 else 'worse'})")

print(f"\n  Output MSE (Y = X @ W.T, fakequant path):")
print(f"    Default ({default_cb.shape[0]}): {output_mse_default:.6f}")
print(f"    Statistical ({stat_cb.shape[0]}): {output_mse_stat:.6f}")
ratio_y = output_mse_stat / output_mse_default
print(f"    Ratio: {ratio_y:.4f}x  ({'better' if ratio_y < 1 else 'worse'})")

print(f"\n{'=' * 70}")
print(f"=== C: Diagnosis ===")
print(f"{'=' * 70}")

has_zero = sum(1 for row in stat_cb.tolist() if 0.0 in row)
print(f"\n  Statistical codebook zero stats:")
print(f"    With 0.0:    {has_zero}/{stat_cb.shape[0]}")
print(f"    Without 0.0: {stat_cb.shape[0] - has_zero}/{stat_cb.shape[0]}")

zero_frac = (blocks == 0.0).float().mean().item()
print(f"  Weight zero fraction: {zero_frac * 100:.1f}%")

print(f"\n  Top-5 statistical codebooks:")
for i in range(min(5, stat_cb.shape[0])):
    vals = [f"{v:+.1f}" for v in stat_cb[i].tolist()]
    print(f"    [{i}] {vals}")

with (output_dir / "model.layers.0.mlp.gate_proj.stats.json").open() as f:
    stats = json.load(f)
print(f"\n  Analysis stats:")
print(f"    Optimality:     {stats.get('optimality_pct', 'N/A')}")
print(f"    Coverage curve: {stats.get('coverage_curve', {})}")

if ratio_eval < 1 and ratio_y > 1:
    print(
        f"\n  >>> MISMATCH: statistical wins in analysis eval but loses in fakequant path"
    )
    print(
        f"  >>> Root cause: _evaluate_codebooks_batch maps raw weights to codebook directly,"
    )
    print(
        f"  >>> but fakequant_blocks_with_scale normalizes -> FP4 -> value_table first."
    )
elif ratio_eval < 1 and ratio_y < 1:
    print(f"\n  >>> Statistical codebook wins in both evaluation methods.")
elif ratio_eval > 1:
    print(f"\n  >>> Statistical codebook loses even in its own analysis evaluation.")
    print(
        f"  >>> Possible cause: fewer entries ({stat_cb.shape[0]} vs {default_cb.shape[0]})."
    )

print()
