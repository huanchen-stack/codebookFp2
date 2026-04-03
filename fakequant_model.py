from __future__ import annotations

# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnusedCallResult=false

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from fakequant import CodebookQuantizer

MLP_SUFFIXES = (".mlp.gate_proj", ".mlp.up_proj", ".mlp.down_proj")
EXPERT_MLP_PATTERN = ".mlp.experts."

CBINT2_COMPUTE_CAP_ENV = "CBINT2_COMPUTE_CAP"
MIN_SM_FOR_FP4 = (8, 9)


def detect_output_format() -> str:
    env_val = os.environ.get(CBINT2_COMPUTE_CAP_ENV)
    if env_val is not None:
        parts = env_val.strip().split(".")
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            cc = (int(parts[0]), int(parts[1]))
        elif len(parts) == 1 and parts[0].isdigit():
            major = int(parts[0])
            cc = (major, 0)
        else:
            raise ValueError(
                f"Invalid {CBINT2_COMPUTE_CAP_ENV}='{env_val}'. "
                "Expected format: '8.0' or '8.9' or '10'"
            )
        return "nvfp4" if cc >= MIN_SM_FOR_FP4 else "bf16"

    if torch.cuda.is_available():
        cc = torch.cuda.get_device_capability()
        return "nvfp4" if cc >= MIN_SM_FOR_FP4 else "bf16"

    return "bf16"


def detect_input_format(weight_map: dict[str, str]) -> str:
    for name in weight_map:
        if name.endswith(".weight_scale"):
            return "nvfp4"
    return "bf16"


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_index(input_path: Path) -> dict[str, str]:
    index_path = input_path / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing index file: {index_path}")

    with index_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    weight_map = data.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError("Invalid index file: missing `weight_map`")

    for key, shard in weight_map.items():
        if not isinstance(key, str) or not isinstance(shard, str):
            raise ValueError("Invalid index file: `weight_map` must map str -> str")

    return weight_map


def _find_quantized_layers(weight_map: dict[str, str]) -> list[str]:
    bases = []
    for name in sorted(weight_map):
        if not name.endswith(".weight"):
            continue
        base = name[: -len(".weight")]
        scale = f"{base}.weight_scale"
        gscale_packed = f"{base}.weight_global_scale"
        gscale_alt = f"{base}.weight_scale_2"
        if scale in weight_map and (gscale_packed in weight_map or gscale_alt in weight_map):
            bases.append(base)
    return bases


def _find_bf16_layers(weight_map: dict[str, str]) -> list[str]:
    bases = []
    for name in sorted(weight_map):
        if not name.endswith(".weight"):
            continue
        base = name[: -len(".weight")]
        scale = f"{base}.weight_scale"
        if scale not in weight_map:
            parts = base.split(".")
            if any(p == "layers" for p in parts):
                bases.append(base)
    return bases


def _is_mlp_layer(base: str) -> bool:
    return any(base.endswith(s) for s in MLP_SUFFIXES) or EXPERT_MLP_PATTERN in base


def _filter_layers(bases: list[str], mlp_only: bool) -> list[str]:
    if not mlp_only:
        return bases
    return [b for b in bases if _is_mlp_layer(b)]


def _weight_key(base: str) -> str:
    return f"{base}.weight"


def _packed_key(base: str) -> str:
    return f"{base}.weight_packed"


def _resolve_weight_name(base: str, weight_map: dict[str, str]) -> str:
    packed = _packed_key(base)
    if packed in weight_map:
        return packed
    return _weight_key(base)


def _resolve_gscale_name(base: str, weight_map: dict[str, str]) -> str:
    gscale_packed = f"{base}.weight_global_scale"
    if gscale_packed in weight_map:
        return gscale_packed
    return f"{base}.weight_scale_2"


def _copy_non_safetensors_files(input_path: Path, output_path: Path) -> None:
    output_path.mkdir(parents=True, exist_ok=True)
    for src in input_path.iterdir():
        if not src.is_file():
            continue
        if src.suffix == ".safetensors":
            continue
        dst = output_path / src.name
        shutil.copy2(src, dst)


def _load_tensor_from_specific_shard(model_path: Path, shard_rel: str, tensor_name: str) -> torch.Tensor:
    shard_path = model_path / shard_rel
    with safe_open(str(shard_path), framework="pt", device="cpu") as f:
        return f.get_tensor(tensor_name)


def _process_shards(
    input_path: Path,
    output_path: Path,
    weight_map: dict[str, str],
    target_layers: list[str],
    quantizer: CodebookQuantizer,
    device: torch.device,
    dry_run: bool,
    vanilla: bool = False,
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

    print(f"Target layers: {len(target_layers)} across {len(shard_order)} shard(s).")
    print(f"Input format: {input_format}  Output format: {output_format}")
    if not dry_run:
        print(f"Using device: {device}")

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
            idx = layer_to_idx[base]

            if dry_run:
                t = tensors[weight_name]
                print(f"  [{idx}/{len(target_layers)}] {base}  dtype={t.dtype}  shape={tuple(t.shape)}")
                continue

            if input_format == "bf16":
                bf16_cpu = tensors[weight_name]
                if output_format == "bf16":
                    print(f"  [{idx}/{len(target_layers)}] bf16→bf16 {base}")
                    quantized = quantizer.fakequant_layer_bf16(bf16_cpu.to(device=device))
                    tensors[weight_name] = quantized.to(device="cpu")
                else:
                    print(f"  [{idx}/{len(target_layers)}] bf16→nvfp4 {base}")
                    w = bf16_cpu.to(device=device, dtype=torch.float32)
                    blocks = w.reshape(-1, 16)
                    opt_fp4, opt_scale = quantizer.fakequant_blocks_with_scale(blocks)
                    out_features, in_features = w.shape
                    opt_fp4 = opt_fp4.reshape(out_features, in_features)
                    new_scale = opt_scale.reshape(out_features, in_features // 16)
                    gscale = torch.tensor([1.0], dtype=torch.float32, device=device)
                    new_weight_scale = quantizer._cast_scale_to_fp8(new_scale / gscale)
                    tensors[weight_name] = quantizer.pack_fp4_to_uint8(opt_fp4).to(device="cpu")
                    tensors[f"{base}.weight_scale"] = new_weight_scale.to(dtype=torch.float8_e4m3fn, device="cpu")
                    tensors[f"{base}.weight_scale_2"] = gscale.to(device="cpu")
            else:
                scale_name = f"{base}.weight_scale"
                gscale_name = _resolve_gscale_name(base, weight_map)
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

                if output_format == "bf16":
                    print(f"  [{idx}/{len(target_layers)}] nvfp4→bf16 {base}")
                    fp4_values = quantizer.unpack_uint8_to_fp4(packed_cpu.to(device=device))
                    gscale = gscale_cpu.to(device=device, dtype=torch.float32).reshape(1)
                    scale_expanded = scale_cpu.to(device=device, dtype=torch.float32).repeat_interleave(16, dim=1)
                    bf16_weights = fp4_values * scale_expanded * gscale
                    quantized = quantizer.fakequant_layer_bf16(bf16_weights)
                    tensors[weight_name] = quantized.to(device="cpu")
                    for drop_key in [scale_name, gscale_name]:
                        tensors.pop(drop_key, None)
                else:
                    mode = "fakequant" if vanilla else "fakequant+scale"
                    print(f"  [{idx}/{len(target_layers)}] {mode} {base}")
                    if vanilla:
                        quantized_packed = quantizer._fakequant_layer_vanilla(
                            packed_cpu.to(device=device),
                            scale_cpu.to(device=device),
                            gscale_cpu.to(device=device),
                        )
                        tensors[weight_name] = quantized_packed.to(device="cpu")
                    else:
                        quantized_packed, new_scale = quantizer.fakequant_layer(
                            packed_cpu.to(device=device),
                            scale_cpu.to(device=device),
                            gscale_cpu.to(device=device),
                        )
                        tensors[weight_name] = quantized_packed.to(device="cpu")
                        tensors[scale_name] = new_scale.to(device="cpu")

        if not dry_run:
            save_file(tensors, str(output_shard))
            print(f"[{shard_idx}/{len(shard_order)}] Saved: {shard_rel}")


def run(input_path: Path, output_path: Path, device: str, mlp_only: bool, dry_run: bool, vanilla: bool = False, output_format: str | None = None) -> None:
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

    if dry_run:
        print("\n=== DRY RUN (no writes) ===\n")
        _process_shards(
            input_path=input_path,
            output_path=output_path,
            weight_map=weight_map,
            target_layers=target_layers,
            quantizer=CodebookQuantizer(),
            device=torch.device("cpu"),
            dry_run=True,
            input_format=input_format,
            output_format=output_format,
        )
        return

    resolved_device = torch.device(device)
    quantizer = CodebookQuantizer()
    start_time = time.perf_counter()

    _copy_non_safetensors_files(input_path, output_path)
    _process_shards(
        input_path=input_path,
        output_path=output_path,
        weight_map=weight_map,
        target_layers=target_layers,
        quantizer=quantizer,
        device=resolved_device,
        dry_run=False,
        vanilla=vanilla,
        input_format=input_format,
        output_format=output_format,
    )

    elapsed = time.perf_counter() - start_time
    print(f"Done. Total elapsed time: {elapsed:.2f}s")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply codebook fake-quantization to quantized or BF16 layers."
    )
    default_input = "models/Qwen3-30B-A3B-NVFP4"
    default_output = default_input + "-CBINT2"
    parser.add_argument("--input-path", type=str, default=default_input)
    parser.add_argument("--output-path", type=str, default=default_output)
    parser.add_argument("--device", type=str, default=_default_device())
    parser.add_argument("--mlp-only", action="store_true",
                        help="Only quantize MLP layers (gate/up/down_proj + MoE experts), skip attention")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print target layer names and shapes without quantizing")
    parser.add_argument("--vanilla", action="store_true",
                        help="Use vanilla codebook selection without scale optimization")
    parser.add_argument("--output-format", type=str, default=None, choices=["nvfp4", "bf16"],
                        help="Output format (default: auto-detect from GPU or CBINT2_COMPUTE_CAP env)")
    args = parser.parse_args()

    run(
        input_path=Path(args.input_path),
        output_path=Path(args.output_path),
        device=args.device,
        mlp_only=args.mlp_only,
        dry_run=args.dry_run,
        vanilla=args.vanilla,
        output_format=args.output_format,
    )


if __name__ == "__main__":
    main()
