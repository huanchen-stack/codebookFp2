from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.multiprocessing as mp
from safetensors.torch import load_file, save_file

from fakequant import CodebookQuantizer
from fakequant_model import (
    _copy_non_safetensors_files,
    _find_quantized_layers,
    _load_index,
    _load_tensor_from_specific_shard,
    _resolve_gscale_name,
    _resolve_weight_name,
)
from gptq.calibrate import layer_block_index


def _group_layers_by_block(target_layers: list[str]) -> dict[int, list[str]]:
    layers_by_block: dict[int, list[str]] = {}
    for name in target_layers:
        idx = layer_block_index(name)
        if idx is not None:
            layers_by_block.setdefault(idx, []).append(name)
    return layers_by_block


def _process_block(
    gpu_id: int,
    block_layers: list[str],
    input_path: Path,
    output_path: Path,
    weight_map: dict[str, str],
) -> None:
    device = torch.device(f"cuda:{gpu_id}")
    unpack = CodebookQuantizer().unpack_uint8_to_fp4

    shard_to_layers: dict[str, list[str]] = {}
    for base in block_layers:
        wn = _resolve_weight_name(base, weight_map)
        shard_to_layers.setdefault(weight_map[wn], []).append(base)

    for shard_rel, layers in shard_to_layers.items():
        output_shard = output_path / shard_rel
        output_shard.parent.mkdir(parents=True, exist_ok=True)
        tensors = load_file(str(input_path / shard_rel), device="cpu")

        for base in layers:
            wn = _resolve_weight_name(base, weight_map)
            scale_name = f"{base}.weight_scale"
            gscale_name = _resolve_gscale_name(base, weight_map)

            packed = tensors[wn].to(device=device)
            scale = tensors.get(scale_name)
            if scale is None:
                scale = _load_tensor_from_specific_shard(
                    input_path, weight_map[scale_name], scale_name
                )
            gscale = tensors.get(gscale_name)
            if gscale is None:
                gscale = _load_tensor_from_specific_shard(
                    input_path, weight_map[gscale_name], gscale_name
                )

            fp4 = unpack(packed)
            scale_exp = scale.to(device=device, dtype=torch.float32).repeat_interleave(
                16, dim=1
            )
            gs = gscale.to(device=device, dtype=torch.float32).reshape(1)
            bf16 = (fp4 * scale_exp * gs).to(dtype=torch.bfloat16)

            tensors[wn] = bf16.cpu()
            for drop in [scale_name, gscale_name, f"{base}.input_scale"]:
                tensors.pop(drop, None)

            print(f"  [GPU {gpu_id}] dequant {base}")

        save_file(tensors, str(output_shard))
        print(f"  [GPU {gpu_id}] Saved: {shard_rel}")


def _gpu_worker(
    gpu_id: int,
    block_indices: list[int],
    layers_by_block: dict[int, list[str]],
    input_path: Path,
    output_path: Path,
    weight_map: dict[str, str],
) -> None:
    torch.cuda.set_device(gpu_id)
    for bi in block_indices:
        _process_block(gpu_id, layers_by_block[bi], input_path, output_path, weight_map)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pure dequant NVFP4 → BF16 (no re-quantization)"
    )
    parser.add_argument(
        "--input-path", type=str, default="/data/models/Qwen3-30B-A3B-NVFP4"
    )
    parser.add_argument(
        "--output-path", type=str, default="/data/models/Qwen3-30B-A3B-NVFP4-BF16"
    )
    parser.add_argument("--num-gpus", type=int, default=8)
    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    weight_map = _load_index(input_path)
    target_layers = _find_quantized_layers(weight_map)

    print(f"Quantized layers: {len(target_layers)}")

    start = time.perf_counter()
    _copy_non_safetensors_files(input_path, output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    layers_by_block = _group_layers_by_block(target_layers)
    sorted_blocks = sorted(layers_by_block.keys())
    effective_gpus = min(
        args.num_gpus, len(sorted_blocks), max(1, torch.cuda.device_count())
    )

    gpu_assignments: list[list[int]] = [[] for _ in range(effective_gpus)]
    for i, bi in enumerate(sorted_blocks):
        gpu_assignments[i % effective_gpus].append(bi)

    print(f"{effective_gpus} GPU(s), {len(sorted_blocks)} blocks")

    ctx = mp.get_context("spawn")
    procs = []
    for gpu_id in range(effective_gpus):
        if not gpu_assignments[gpu_id]:
            continue
        p = ctx.Process(
            target=_gpu_worker,
            args=(
                gpu_id,
                gpu_assignments[gpu_id],
                layers_by_block,
                input_path,
                output_path,
                weight_map,
            ),
        )
        procs.append(p)
        p.start()

    for p in procs:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"Worker {p.name} failed with exit code {p.exitcode}")

    block_shards = set()
    for bi in sorted_blocks:
        for base in layers_by_block[bi]:
            block_shards.add(weight_map[_resolve_weight_name(base, weight_map)])
    for sf in set(weight_map.values()) - block_shards:
        out = output_path / sf
        if not out.exists():
            save_file(load_file(str(input_path / sf), device="cpu"), str(out))
            print(f"Copied: {sf}")

    print(f"Done. {time.perf_counter() - start:.1f}s")


if __name__ == "__main__":
    main()
