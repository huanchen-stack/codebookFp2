import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from safetensors.torch import save_file, load_file

from fakequant import CodebookQuantizer

torch.manual_seed(0)
q = CodebookQuantizer()

FP8_MAX = 448.0
FP4_MAX = 6.0
OUT, IN = 2048, 768

bf16_weights = torch.randn(OUT, IN, dtype=torch.float32) * 0.5

abs_max = bf16_weights.abs().max().item()
weight_scale_2 = torch.tensor(abs_max / (FP8_MAX * FP4_MAX), dtype=torch.float32)
actual_gs = FP8_MAX * FP4_MAX / abs_max

raw_per_group = bf16_weights.reshape(OUT, IN // 16, 16).abs().amax(dim=-1) / FP4_MAX
weight_scale = (raw_per_group * actual_gs).to(torch.float8_e4m3fn)

eff_scale = weight_scale.to(torch.float32) * weight_scale_2.item()
eff_expanded = eff_scale.repeat_interleave(16, dim=1)
fp4_idx = (bf16_weights / eff_expanded).unsqueeze(-1).sub(q.fp4_representable).abs().argmin(dim=-1)
fp4_values = q.fp4_representable[fp4_idx]
weight_packed = q.pack_fp4_to_uint8(fp4_values)

dequant_orig = fp4_values * eff_expanded
nvfp4_mse = ((bf16_weights - dequant_orig) ** 2).mean().item()

print(f"=== Synthetic NVFP4 layer ({OUT}x{IN}) ===")
print(f"  weight_scale_2:  {weight_scale_2.item():.6f}  (dtype={weight_scale_2.dtype}, shape={weight_scale_2.shape})")
print(f"  weight_scale:    dtype={weight_scale.dtype}  shape={weight_scale.shape}")
print(f"  weight_packed:   dtype={weight_packed.dtype}  shape={weight_packed.shape}")
print(f"  NVFP4 MSE:       {nvfp4_mse:.6f}")

with tempfile.TemporaryDirectory() as tmpdir:
    input_dir = Path(tmpdir) / "input"
    input_dir.mkdir()

    layer_base = "model.layers.0.mlp.gate_proj"
    tensors = {
        f"{layer_base}.weight": weight_packed,
        f"{layer_base}.weight_scale": weight_scale,
        f"{layer_base}.weight_scale_2": weight_scale_2,
    }
    shard_name = "model-00001-of-00001.safetensors"
    save_file(tensors, str(input_dir / shard_name))

    index = {
        "metadata": {},
        "weight_map": {k: shard_name for k in tensors},
    }
    with (input_dir / "model.safetensors.index.json").open("w") as f:
        json.dump(index, f)

    for mode_name, flag in [("vanilla", "--vanilla"), ("scale-opt", "")]:
        output_dir = Path(tmpdir) / f"output-{mode_name}"
        cmd = [
            sys.executable, "fakequant_model.py",
            "--input-path", str(input_dir),
            "--output-path", str(output_dir),
            "--device", "cpu",
        ]
        if flag:
            cmd.append(flag)

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"\n  [{mode_name}] FAILED:")
            print(result.stderr)
            continue

        out_tensors = load_file(str(output_dir / shard_name))
        out_packed = out_tensors[f"{layer_base}.weight"]
        out_scale = out_tensors[f"{layer_base}.weight_scale"]
        out_gs = out_tensors[f"{layer_base}.weight_scale_2"]

        out_fp4 = q.unpack_uint8_to_fp4(out_packed)
        out_eff = out_scale.to(torch.float32) * out_gs.to(torch.float32)
        out_eff_expanded = out_eff.repeat_interleave(16, dim=1)
        dequant_out = out_fp4 * out_eff_expanded

        mse = ((bf16_weights - dequant_out) ** 2).mean().item()

        print(f"\n  [{mode_name}]")
        print(f"    weight dtype={out_packed.dtype}  shape={out_packed.shape}")
        print(f"    scale  dtype={out_scale.dtype}  shape={out_scale.shape}")
        print(f"    gs     dtype={out_gs.dtype}  shape={out_gs.shape}  value={out_gs.float().item():.6f}")
        print(f"    dequant MSE:   {mse:.6f}  ({mse / nvfp4_mse:.2f}x vs NVFP4)")

        assert out_packed.dtype == torch.uint8, f"weight dtype wrong: {out_packed.dtype}"
        assert out_packed.shape == weight_packed.shape, f"weight shape wrong: {out_packed.shape}"
        assert out_scale.dtype == weight_scale.dtype, f"scale dtype changed: {weight_scale.dtype} -> {out_scale.dtype}"
        assert out_scale.shape == weight_scale.shape, f"scale shape changed: {weight_scale.shape} -> {out_scale.shape}"
        assert out_gs.float().item() == weight_scale_2.float().item(), \
            f"weight_scale_2 changed: {weight_scale_2.item()} -> {out_gs.item()}"

        blocks_out = out_fp4.reshape(-1, 16)
        max_distinct = max(len(set(blocks_out[i].tolist())) for i in range(blocks_out.shape[0]))
        assert max_distinct <= 4, f"block has {max_distinct} distinct values"

        dequant_err = (dequant_out - bf16_weights).abs().max().item()
        has_nan = dequant_out.isnan().any().item()
        print(f"    max |error|:   {dequant_err:.4f}  NaN: {has_nan}")
        if has_nan:
            nan_scales = out_scale.float().isnan().sum().item()
            zero_scales = (out_scale.float() == 0).sum().item()
            neg_scales = (out_scale.float() < 0).sum().item()
            print(f"    scale NaN: {nan_scales}  zeros: {zero_scales}  negative: {neg_scales}")
            print(f"    FAIL (NaN in dequant)")
        else:
            print(f"    PASS")

print("\n=== Direct-call scale-opt: no NaN ===")
new_packed_d, new_ws_d = q.fakequant_layer(weight_packed, weight_scale, weight_scale_2)
new_fp4_d = q.unpack_uint8_to_fp4(new_packed_d)
new_eff_d = new_ws_d.to(torch.float32) * weight_scale_2.to(torch.float32)
new_eff_d_exp = new_eff_d.repeat_interleave(16, dim=1)
dequant_d = new_fp4_d * new_eff_d_exp
assert not dequant_d.isnan().any(), "NaN in direct-call dequant"
assert not new_ws_d.float().isnan().any(), "NaN in direct-call scale"
direct_mse = ((bf16_weights - dequant_d) ** 2).mean().item()
print(f"  MSE: {direct_mse:.6f}  NaN: False")
print(f"  PASS")

print("\n=== BF16 input model: fakequant_model.py --output-format bf16 ===")
with tempfile.TemporaryDirectory() as tmpdir_bf16:
    bf16_input_dir = Path(tmpdir_bf16) / "bf16_input"
    bf16_input_dir.mkdir()

    torch.manual_seed(42)
    bf16_raw = torch.randn(OUT, IN, dtype=torch.bfloat16)
    layer_base_bf16 = "model.layers.0.mlp.gate_proj"
    bf16_tensors = {
        f"{layer_base_bf16}.weight": bf16_raw,
    }
    bf16_shard = "model-00001-of-00001.safetensors"
    save_file(bf16_tensors, str(bf16_input_dir / bf16_shard))

    bf16_index = {
        "metadata": {},
        "weight_map": {k: bf16_shard for k in bf16_tensors},
    }
    with (bf16_input_dir / "model.safetensors.index.json").open("w") as f:
        json.dump(bf16_index, f)

    bf16_output_dir = Path(tmpdir_bf16) / "bf16_output"
    cmd_bf16 = [
        sys.executable, "fakequant_model.py",
        "--input-path", str(bf16_input_dir),
        "--output-path", str(bf16_output_dir),
        "--device", "cpu",
        "--output-format", "bf16",
    ]
    result_bf16 = subprocess.run(cmd_bf16, capture_output=True, text=True)
    if result_bf16.returncode != 0:
        print(f"  FAILED:")
        print(result_bf16.stderr)
    else:
        out_bf16_tensors = load_file(str(bf16_output_dir / bf16_shard))
        out_bf16_w = out_bf16_tensors[f"{layer_base_bf16}.weight"]

        assert out_bf16_w.shape == bf16_raw.shape, f"Shape mismatch: {out_bf16_w.shape} vs {bf16_raw.shape}"
        assert f"{layer_base_bf16}.weight_scale" not in out_bf16_tensors, "BF16 output should not have weight_scale"
        assert not out_bf16_w.isnan().any(), "NaN in BF16 output"

        bf16_mse = ((bf16_raw.float() - out_bf16_w.float()) ** 2).mean().item()
        print(f"  output dtype={out_bf16_w.dtype}  shape={out_bf16_w.shape}")
        print(f"  MSE vs original: {bf16_mse:.6f}")
        print(f"  PASS")

    print("\n=== BF16 input model: fakequant_model.py --output-format nvfp4 ===")
    nvfp4_output_dir = Path(tmpdir_bf16) / "nvfp4_output"
    cmd_nvfp4 = [
        sys.executable, "fakequant_model.py",
        "--input-path", str(bf16_input_dir),
        "--output-path", str(nvfp4_output_dir),
        "--device", "cpu",
        "--output-format", "nvfp4",
    ]
    result_nvfp4 = subprocess.run(cmd_nvfp4, capture_output=True, text=True)
    if result_nvfp4.returncode != 0:
        print(f"  FAILED:")
        print(result_nvfp4.stderr)
    else:
        out_nvfp4_tensors = load_file(str(nvfp4_output_dir / bf16_shard))
        out_nvfp4_packed = out_nvfp4_tensors[f"{layer_base_bf16}.weight"]
        out_nvfp4_scale = out_nvfp4_tensors[f"{layer_base_bf16}.weight_scale"]
        out_nvfp4_gs = out_nvfp4_tensors[f"{layer_base_bf16}.weight_scale_2"]

        assert out_nvfp4_packed.dtype == torch.uint8, f"Expected uint8 packed, got {out_nvfp4_packed.dtype}"
        assert out_nvfp4_packed.shape == (OUT, IN // 2), f"Packed shape mismatch: {out_nvfp4_packed.shape}"

        out_fp4 = q.unpack_uint8_to_fp4(out_nvfp4_packed)
        out_eff = out_nvfp4_scale.to(torch.float32) * out_nvfp4_gs.to(torch.float32)
        out_eff_exp = out_eff.repeat_interleave(16, dim=1)
        dequant_nvfp4_from_bf16 = out_fp4 * out_eff_exp
        nvfp4_from_bf16_mse = ((bf16_raw.float() - dequant_nvfp4_from_bf16) ** 2).mean().item()

        assert not dequant_nvfp4_from_bf16.isnan().any(), "NaN in nvfp4-from-bf16 dequant"
        print(f"  packed dtype={out_nvfp4_packed.dtype}  shape={out_nvfp4_packed.shape}")
        print(f"  scale  dtype={out_nvfp4_scale.dtype}  shape={out_nvfp4_scale.shape}")
        print(f"  MSE vs original: {nvfp4_from_bf16_mse:.6f}")
        print(f"  PASS")

print("\nAll test_fakequant_model.py tests passed.")
