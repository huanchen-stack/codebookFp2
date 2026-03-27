from __future__ import annotations

# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportAny=false, reportUnknownArgumentType=false, reportUnusedCallResult=false

import argparse
import json
import shutil
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from fakequant import CodebookQuantizer


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
    quantizer: CodebookQuantizer,
    device: torch.device,
) -> None:
    shard_order: list[str] = []
    seen: set[str] = set()
    for shard in weight_map.values():
        if shard not in seen:
            seen.add(shard)
            shard_order.append(shard)

    quantized_keys = sorted(name for name in weight_map if name.endswith(".weight_packed"))
    total_layers = len(quantized_keys)
    if total_layers == 0:
        raise RuntimeError("No quantized layers found (no `*.weight_packed` tensors).")

    layer_to_idx = {name: idx + 1 for idx, name in enumerate(quantized_keys)}

    print(f"Found {total_layers} quantized layers across {len(shard_order)} shard(s).")
    print(f"Using device: {device}")

    for shard_idx, shard_rel in enumerate(shard_order, start=1):
        input_shard = input_path / shard_rel
        output_shard = output_path / shard_rel
        output_shard.parent.mkdir(parents=True, exist_ok=True)

        print(f"[{shard_idx}/{len(shard_order)}] Loading shard: {shard_rel}")
        tensors = load_file(str(input_shard), device="cpu")

        shard_quantized = sorted(name for name in tensors if name.endswith(".weight_packed"))

        for packed_name in shard_quantized:
            base = packed_name[: -len(".weight_packed")]
            scale_name = f"{base}.weight_scale"
            gscale_name = f"{base}.weight_global_scale"

            if scale_name not in weight_map or gscale_name not in weight_map:
                raise KeyError(
                    f"Missing scale tensors for layer `{base}`; expected `{scale_name}` and `{gscale_name}`"
                )

            packed_cpu = tensors[packed_name]

            if scale_name in tensors:
                scale_cpu = tensors[scale_name]
            else:
                scale_cpu = _load_tensor_from_specific_shard(
                    model_path=input_path,
                    shard_rel=weight_map[scale_name],
                    tensor_name=scale_name,
                )

            if gscale_name in tensors:
                gscale_cpu = tensors[gscale_name]
            else:
                gscale_cpu = _load_tensor_from_specific_shard(
                    model_path=input_path,
                    shard_rel=weight_map[gscale_name],
                    tensor_name=gscale_name,
                )

            layer_idx = layer_to_idx[packed_name]
            print(f"  - [{layer_idx}/{total_layers}] fakequant {base}")

            packed_dev = packed_cpu.to(device=device)
            scale_dev = scale_cpu.to(device=device)
            gscale_dev = gscale_cpu.to(device=device)

            quantized_packed = quantizer.fakequant_layer(
                packed_dev,
                scale_dev,
                gscale_dev,
            )

            tensors[packed_name] = quantized_packed.to(device="cpu")

        save_file(tensors, str(output_shard))
        print(f"[{shard_idx}/{len(shard_order)}] Saved shard: {shard_rel}")


def run(input_path: Path, output_path: Path, device: str) -> None:
    if not input_path.exists() or not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist or is not a directory: {input_path}")

    resolved_device = torch.device(device)
    quantizer = CodebookQuantizer()

    start_time = time.perf_counter()

    weight_map = _load_index(input_path)
    _copy_non_safetensors_files(input_path, output_path)

    _process_shards(
        input_path=input_path,
        output_path=output_path,
        weight_map=weight_map,
        quantizer=quantizer,
        device=resolved_device,
    )

    elapsed = time.perf_counter() - start_time
    print(f"Done. Total elapsed time: {elapsed:.2f}s")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply codebook fake-quantization to all NVFP4 quantized layers."
    )
    parser.add_argument("--input-path", type=str, required=True, help="Input NVFP4 model directory")
    parser.add_argument("--output-path", type=str, required=True, help="Output model directory")
    parser.add_argument(
        "--device",
        type=str,
        default=_default_device(),
        help='Device for fakequant processing (default: "cuda" if available else "cpu")',
    )
    args = parser.parse_args()

    run(
        input_path=Path(args.input_path),
        output_path=Path(args.output_path),
        device=args.device,
    )


if __name__ == "__main__":
    main()
