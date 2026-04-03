from __future__ import annotations

# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnusedCallResult=false

import argparse
import time
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from fakequant import CodebookQuantizer
from fakequant_model import (
    _copy_non_safetensors_files,
    _default_device,
    _filter_layers,
    _find_bf16_layers,
    _find_quantized_layers,
    _load_index,
    _load_tensor_from_specific_shard,
    _resolve_gscale_name,
    _resolve_weight_name,
    detect_input_format,
    detect_output_format,
)
from gptq import CodebookGPTQ
from gptq.calibrate import hessian_block_keys, layer_block_index, load_hessian


def _format_missing_layers(missing_layers: list[str], artifact_label: str, artifact_dir: Path) -> str:
    preview = ", ".join(missing_layers[:3])
    suffix = "" if len(missing_layers) <= 3 else ", ..."
    return (
        f"Missing {len(missing_layers)} {artifact_label} in {artifact_dir}. "
        f"Examples: {preview}{suffix}"
    )


def _group_layers_by_block(target_layers: list[str]) -> dict[int, list[str]]:
    layers_by_block: dict[int, list[str]] = {}
    for layer_name in target_layers:
        block_idx = layer_block_index(layer_name)
        if block_idx is None:
            raise ValueError(f"Target layer is not inside a transformer block: {layer_name}")
        layers_by_block.setdefault(block_idx, []).append(layer_name)
    return layers_by_block


def _validate_local_calibration_artifacts(
    artifact_dir: Path,
    target_layers: list[str],
) -> None:
    missing_hessians: list[str] = []
    for block_idx, layer_names in _group_layers_by_block(target_layers).items():
        available_keys = hessian_block_keys(artifact_dir, block_idx)
        missing_hessians.extend(layer_name for layer_name in layer_names if layer_name not in available_keys)

    if missing_hessians:
        raise FileNotFoundError(
            _format_missing_layers(missing_hessians, "Hessian entries", artifact_dir)
            + ". Run `python -m gptq.calibrate --output-dir ...` first."
        )


def _process_shards_gptq(
    input_path: Path,
    output_path: Path,
    weight_map: dict[str, str],
    target_layers: list[str],
    quantizer: CodebookQuantizer,
    device: torch.device,
    calibration_dir: Path,
    dry_run: bool,
    input_format: str = "nvfp4",
    output_format: str = "nvfp4",
) -> None:
    shard_order: list[str] = []
    seen: set[str] = set()
    for shard in weight_map.values():
        if shard not in seen:
            seen.add(shard)
            shard_order.append(shard)

    layer_to_idx = {b: idx + 1 for idx, b in enumerate(target_layers)}

    print(f"\nTarget layers: {len(target_layers)} across {len(shard_order)} shard(s).")
    print(f"Input format: {input_format}  Output format: {output_format}")
    if not dry_run:
        print(f"Using device: {device}")
        print(f"Calibration dir: {calibration_dir}")

    block_key_cache: dict[int, set[str]] = {}

    for shard_idx, shard_rel in enumerate(shard_order, start=1):
        input_shard = input_path / shard_rel
        output_shard = output_path / shard_rel
        output_shard.parent.mkdir(parents=True, exist_ok=True)

        tensors = load_file(str(input_shard), device="cpu")
        shard_targets = sorted(
            b for b in target_layers
            if _resolve_weight_name(b, weight_map) in tensors
        )

        if not shard_targets:
            if not dry_run:
                save_file(tensors, str(output_shard))
            continue

        print(f"[{shard_idx}/{len(shard_order)}] {shard_rel}  ({len(shard_targets)} target layers)")

        for base in shard_targets:
            weight_name = _resolve_weight_name(base, weight_map)
            scale_name = f"{base}.weight_scale"
            gscale_name = _resolve_gscale_name(base, weight_map)
            idx = layer_to_idx[base]

            if dry_run:
                t = tensors[weight_name]
                block_idx = layer_block_index(base)
                if block_idx is None:
                    raise ValueError(f"Target layer is not inside a transformer block: {base}")
                if block_idx not in block_key_cache:
                    block_key_cache[block_idx] = hessian_block_keys(calibration_dir, block_idx)
                h_status = "✓" if base in block_key_cache[block_idx] else "✗"
                print(f"  [{idx}/{len(target_layers)}] {base}  shape={tuple(t.shape)}  hessian={h_status}")
                continue

            gscale_for_output = torch.tensor([1.0], device=device)
            if input_format == "bf16":
                w = tensors[weight_name].to(device=device, dtype=torch.float32)
                _, in_features = w.shape
                ref_mode = "bf16"
            else:
                packed_cpu = tensors[weight_name]
                scale_cpu = tensors.get(scale_name)
                if scale_cpu is None:
                    scale_cpu = _load_tensor_from_specific_shard(
                        input_path, weight_map[scale_name], scale_name
                    )
                gscale_cpu = tensors.get(gscale_name)
                if gscale_cpu is None:
                    gscale_cpu = _load_tensor_from_specific_shard(
                        input_path, weight_map[gscale_name], gscale_name
                    )
                packed = packed_cpu.to(device=device)
                scale = scale_cpu.to(device=device)
                gscale_for_output = gscale_cpu.to(device=device, dtype=torch.float32).reshape(1)
                fp4_values = quantizer.unpack_uint8_to_fp4(packed)
                _, in_features = fp4_values.shape
                scale_expanded = scale.to(torch.float32).repeat_interleave(16, dim=1)
                w = fp4_values * scale_expanded * gscale_for_output
                ref_mode = "nvfp4"

            gptq = CodebookGPTQ(in_features=in_features, quantizer=quantizer)

            hessian = load_hessian(calibration_dir, base)
            if hessian is None:
                raise FileNotFoundError(f"Missing Hessian for {base} in {calibration_dir}")
            gptq.H = hessian.to(device)
            gptq.num_samples = 1

            print(f"  [{idx}/{len(target_layers)}] gptq({ref_mode})→{output_format} {base}")

            fp4_out, scales_out, _ = gptq.quantize(w)

            if output_format == "bf16":
                dequantized = (fp4_out * scales_out).to(dtype=torch.bfloat16)
                tensors[weight_name] = dequantized.to(device="cpu")
                for drop_key in [scale_name, gscale_name]:
                    tensors.pop(drop_key, None)
            else:
                gscale_val = gscale_for_output
                new_packed = quantizer.pack_fp4_to_uint8(fp4_out)
                new_weight_scale = quantizer._cast_scale_to_fp8(scales_out / gscale_val)
                tensors[weight_name] = new_packed.to(device="cpu")
                tensors[scale_name] = new_weight_scale.to(dtype=torch.float8_e4m3fn, device="cpu")
                if input_format == "bf16":
                    tensors[f"{base}.weight_scale_2"] = gscale_val.to(device="cpu")

            del gptq, w, hessian
            if device.type == "cuda":
                torch.cuda.empty_cache()

        if not dry_run:
            save_file(tensors, str(output_shard))
            print(f"[{shard_idx}/{len(shard_order)}] Saved: {shard_rel}")


def run(
    input_path: Path,
    output_path: Path,
    device: str,
    mlp_only: bool,
    dry_run: bool,
    hessian_dir: str = "hessians",
    output_format: str | None = None,
) -> None:
    if not input_path.exists() or not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist or is not a directory: {input_path}")

    weight_map = _load_index(input_path)
    input_format = detect_input_format(weight_map)
    if output_format is None:
        output_format = detect_output_format()

    if input_format == "nvfp4":
        all_layers = _find_quantized_layers(weight_map)
    else:
        all_layers = _find_bf16_layers(weight_map)
    target_layers = _filter_layers(all_layers, mlp_only)

    print(f"Input format: {input_format}  Output format: {output_format}")
    print(f"Layers in model: {len(all_layers)}")
    print(f"Selected for processing: {len(target_layers)}")
    if mlp_only:
        print(f"Skipped (non-MLP): {len(all_layers) - len(target_layers)}")

    hessian_path = Path(hessian_dir)
    if not hessian_path.exists() or not hessian_path.is_dir():
        raise FileNotFoundError(
            f"Calibration directory does not exist or is not a directory: {hessian_path}. "
            + "Run `python -m gptq.calibrate --output-dir ...` first."
        )
    _validate_local_calibration_artifacts(hessian_path, target_layers)

    if dry_run:
        print("\n=== DRY RUN (no writes) ===\n")
        _process_shards_gptq(
            input_path=input_path,
            output_path=output_path,
            weight_map=weight_map,
            target_layers=target_layers,
            quantizer=CodebookQuantizer(),
            device=torch.device("cpu"),
            calibration_dir=hessian_path,
            dry_run=True,
            input_format=input_format,
            output_format=output_format,
        )
        return

    resolved_device = torch.device(device)
    quantizer = CodebookQuantizer()
    start_time = time.perf_counter()

    _copy_non_safetensors_files(input_path, output_path)
    _process_shards_gptq(
        input_path=input_path,
        output_path=output_path,
        weight_map=weight_map,
        target_layers=target_layers,
        quantizer=quantizer,
        device=resolved_device,
        calibration_dir=hessian_path,
        dry_run=False,
        input_format=input_format,
        output_format=output_format,
    )

    elapsed = time.perf_counter() - start_time
    print(f"Done. Total elapsed time: {elapsed:.2f}s")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply GPTQ-optimized codebook quantization to NVFP4 or BF16 layers."
    )
    default_input = "models/Qwen3-30B-A3B-NVFP4"
    default_output = default_input + "-CBINT2-GPTQ"
    parser.add_argument("--input-path", type=str, default=default_input)
    parser.add_argument("--output-path", type=str, default=default_output)
    parser.add_argument("--device", type=str, default=_default_device())
    parser.add_argument("--mlp-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--hessian-dir", type=str, required=True,
                        help="Local calibration directory produced by `python -m gptq.calibrate --output-dir ...`")
    parser.add_argument("--output-format", type=str, default=None, choices=["nvfp4", "bf16"],
                        help="Output format (default: auto-detect from GPU or CBINT2_COMPUTE_CAP env)")
    args = parser.parse_args()

    run(
        input_path=Path(args.input_path),
        output_path=Path(args.output_path),
        device=args.device,
        mlp_only=args.mlp_only,
        dry_run=args.dry_run,
        hessian_dir=args.hessian_dir,
        output_format=args.output_format,
    )


if __name__ == "__main__":
    main()
