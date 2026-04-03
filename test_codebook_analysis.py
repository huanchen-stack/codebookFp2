import json
import sys
import tempfile
from pathlib import Path

import torch
from safetensors.torch import save_file

from codebook_analysis import run_analysis
from fakequant import CodebookQuantizer

torch.manual_seed(42)

OUT, IN = 64, 256

tmpdir = tempfile.mkdtemp()
model_dir = Path(tmpdir) / "model"
hessian_dir = Path(tmpdir) / "hessians"
model_dir.mkdir()
hessian_dir.mkdir()

weight_map = {}
for block_idx in range(2):
    shard_file = f"block_{block_idx:04d}.safetensors"
    layer_names = [
        f"model.layers.{block_idx}.mlp.gate_proj",
        f"model.layers.{block_idx}.mlp.up_proj",
        f"model.layers.{block_idx}.mlp.down_proj",
    ]
    tensors = {}
    for ln in layer_names:
        tensors[f"{ln}.weight"] = torch.randn(OUT, IN, dtype=torch.bfloat16) * 0.3
        weight_map[f"{ln}.weight"] = shard_file
    save_file(tensors, str(model_dir / shard_file))

    h_tensors = {}
    for ln in layer_names:
        H = torch.eye(IN, dtype=torch.float32) * 0.25
        for ch in [3, 7, 19, 35]:
            H[ch, ch] = 12.5
        h_tensors[ln] = H
    save_file(h_tensors, str(hessian_dir / f"block_{block_idx:02d}.safetensors"))

index = {"metadata": {}, "weight_map": weight_map}
with (model_dir / "model.safetensors.index.json").open("w") as f:
    json.dump(index, f)

print("=== Frequency method ===")
freq_dir = Path(tmpdir) / "cb_freq"
run_analysis(
    model_path=model_dir, hessian_dir=hessian_dir, output_dir=freq_dir,
    mlp_only=True, num_codebooks=32, selection_method="frequency",
    coverage_threshold=1.05, device_str="cpu", chunk_size=512,
)

print("\n=== Greedy method ===")
greedy_dir = Path(tmpdir) / "cb_greedy"
run_analysis(
    model_path=model_dir, hessian_dir=hessian_dir, output_dir=greedy_dir,
    mlp_only=True, num_codebooks=32, selection_method="greedy",
    coverage_threshold=1.05, device_str="cpu", chunk_size=512,
)

print("\n=== Per-layer file verification ===")
import glob
freq_files = sorted(glob.glob(str(freq_dir / "*.pt")))
greedy_files = sorted(glob.glob(str(greedy_dir / "*.pt")))
assert len(freq_files) == 6, f"Expected 6 freq .pt files, got {len(freq_files)}"
assert len(greedy_files) == 6, f"Expected 6 greedy .pt files, got {len(greedy_files)}"
print(f"  Frequency: {len(freq_files)} .pt files")
print(f"  Greedy:    {len(greedy_files)} .pt files")

for f in freq_files:
    cb = torch.load(f, weights_only=True)
    assert cb.shape == (32, 4), f"{Path(f).name} has shape {tuple(cb.shape)}, expected (32, 4)"
print(f"  All shapes: (32, 4)  PASS")

print("\n=== Summary JSON verification ===")
with open(str(freq_dir / "codebook_summary.json")) as f:
    summary = json.load(f)
assert len(summary["layers"]) == 6
assert "global_stats" in summary
assert "coverage_curve" in summary["global_stats"]
print(f"  Layers: {len(summary['layers'])}")
print(f"  Coverage curve: {summary['global_stats']['coverage_curve']}")
print(f"  PASS")

print("\n=== Cross-layer codebook comparison ===")
layer_codebooks = {}
for f in sorted(freq_files):
    name = Path(f).stem
    layer_codebooks[name] = torch.load(f, weights_only=True)

layer_names_sorted = sorted(layer_codebooks.keys())
print(f"  Comparing {len(layer_names_sorted)} layers:")

all_same = True
for i in range(len(layer_names_sorted)):
    for j in range(i + 1, len(layer_names_sorted)):
        name_i = layer_names_sorted[i]
        name_j = layer_names_sorted[j]
        cb_i = layer_codebooks[name_i]
        cb_j = layer_codebooks[name_j]

        set_i = set(tuple(row.tolist()) for row in cb_i)
        set_j = set(tuple(row.tolist()) for row in cb_j)

        shared = len(set_i & set_j)
        only_i = len(set_i - set_j)
        only_j = len(set_j - set_i)
        overlap_pct = shared / len(set_i) * 100

        if shared != len(set_i) or shared != len(set_j):
            all_same = False

        if i == 0 or overlap_pct < 90:
            short_i = name_i.split(".")[-1] if "." in name_i else name_i
            short_j = name_j.split(".")[-1] if "." in name_j else name_j
            print(f"    {short_i} vs {short_j}: {shared} shared, {only_i}+{only_j} unique ({overlap_pct:.0f}% overlap)")

if all_same:
    print(f"  Result: ALL layers have IDENTICAL codebooks")
else:
    print(f"  Result: Layers have DIFFERENT codebooks (per-layer specialization confirmed)")

print("\n=== set_codebook() verification ===")
q = CodebookQuantizer()
assert q.codebook.shape == (364, 4)
assert q._value_table.shape == (15, 364)

layer_cb = torch.load(freq_files[0], weights_only=True)
q.set_codebook(layer_cb)
assert q.codebook.shape == (32, 4)
assert q._value_table.shape == (15, 32)

test_blocks = torch.randn(64, 16)
q_fp4, s_opt = q.fakequant_blocks_with_scale(test_blocks)
assert q_fp4.shape == (64, 16)
assert s_opt.shape == (64, 1)
assert not q_fp4.isnan().any()

for i in range(q_fp4.shape[0]):
    distinct = len(set(q_fp4[i].tolist()))
    assert distinct <= 4, f"Block {i} has {distinct} distinct values after set_codebook"
print(f"  Original: (364, 4) → set_codebook → (32, 4)")
print(f"  fakequant_blocks_with_scale: max 4 distinct per block  PASS")

q.set_codebook(torch.load(freq_files[-1], weights_only=True))
q_fp4_2, _ = q.fakequant_blocks_with_scale(test_blocks)
for i in range(q_fp4_2.shape[0]):
    assert len(set(q_fp4_2[i].tolist())) <= 4
print(f"  Second set_codebook swap: PASS")

print("\nAll test_codebook_analysis.py tests passed.")
