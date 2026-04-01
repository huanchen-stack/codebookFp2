from __future__ import annotations

# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnusedCallResult=false

import argparse
import json
import shutil
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from fakequant import CodebookQuantizer

MLP_SUFFIXES = (".mlp.gate_proj", ".mlp.up_proj", ".mlp.down_proj")
EXPERT_MLP_PATTERN = ".mlp.experts."


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
) -> None:
    shard_order: list[str] = []
    seen: set[str] = set()
    for shard in weight_map.values():
        if shard not in seen:
            seen.add(shard)
            shard_order.append(shard)

    target_weight_names = {_resolve_weight_name(b, weight_map) for b in target_layers}
    layer_to_idx = {b: idx + 1 for idx, b in enumerate(target_layers)}

    print(f"Target layers: {len(target_layers)} across {len(shard_order)} shard(s).")
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
            scale_name = f"{base}.weight_scale"
            gscale_name = _resolve_gscale_name(base, weight_map)

            idx = layer_to_idx[base]

            if dry_run:
                t = tensors[weight_name]
                print(f"  [{idx}/{len(target_layers)}] {base}  dtype={t.dtype}  shape={tuple(t.shape)}")
                continue

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


def run(input_path: Path, output_path: Path, device: str, mlp_only: bool, dry_run: bool, vanilla: bool = False) -> None:
    if not input_path.exists() or not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist or is not a directory: {input_path}")

    weight_map = _load_index(input_path)
    all_layers = _find_quantized_layers(weight_map)
    target_layers = _filter_layers(all_layers, mlp_only)

    print(f"Quantized layers in model: {len(all_layers)}")
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
    )

    elapsed = time.perf_counter() - start_time
    print(f"Done. Total elapsed time: {elapsed:.2f}s")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply codebook fake-quantization to NVFP4 quantized layers."
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
    args = parser.parse_args()

    run(
        input_path=Path(args.input_path),
        output_path=Path(args.output_path),
        device=args.device,
        mlp_only=args.mlp_only,
        dry_run=args.dry_run,
        vanilla=args.vanilla,
    )


if __name__ == "__main__":
    main()
