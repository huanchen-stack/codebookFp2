from __future__ import annotations

import argparse
import itertools
import json
import re
import time
from pathlib import Path

import torch
import torch.multiprocessing as mp
from safetensors import safe_open
from safetensors.torch import load_file

from fakequant import CodebookQuantizer
from fakequant_model import (
    _filter_layers,
    _find_bf16_layers,
    _find_quantized_layers,
    _load_index,
    _resolve_gscale_name,
    _resolve_weight_name,
    detect_input_format,
)
from gptq.calibrate import hessian_block_file, layer_block_index


FP4_ALL_VALUES = [
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
]


def _build_all_candidate_codebooks() -> torch.Tensor:
    combos = list(itertools.combinations(FP4_ALL_VALUES, 4))
    return torch.tensor(combos, dtype=torch.float32)


def _cast_scale_to_fp8(scale: torch.Tensor) -> torch.Tensor:
    return (
        scale.clamp(-448.0, 448.0)
        .to(torch.float8_e4m3fn)
        .to(torch.float32)
        .clamp(min=1e-10)
    )


def _evaluate_codebooks_batch(
    blocks: torch.Tensor,
    importance: torch.Tensor,
    all_codebooks: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    C = all_codebooks.shape[0]
    device = blocks.device

    imp = importance / importance.mean(dim=-1, keepdim=True).clamp(min=1e-10)

    dists = (blocks[:, :, None, None] - all_codebooks[None, None, :, :]).abs()
    nearest_idx = dists.argmin(dim=-1)
    del dists

    k_range = torch.arange(C, device=device)
    q_mapped = all_codebooks[k_range, nearest_idx]
    del nearest_idx

    imp_3d = imp.unsqueeze(-1)
    numer = (imp_3d * blocks.unsqueeze(-1) * q_mapped).sum(dim=1)
    denom = (imp_3d * q_mapped**2).sum(dim=1).clamp(min=1e-10)
    del q_mapped

    s_all = _cast_scale_to_fp8(numer / denom)
    w_sq = (imp * blocks**2).sum(dim=1, keepdim=True)
    mse = w_sq - 2 * s_all * numer + s_all**2 * denom
    mse[numer <= 0] = float("inf")

    best_k = mse.argmin(dim=-1)
    return best_k, mse


def _select_frequency(
    winners: torch.Tensor,
    num_codebooks: int,
    total_candidates: int,
) -> torch.Tensor:
    freq = torch.zeros(total_candidates, dtype=torch.long)
    for idx in winners.cpu().tolist():
        freq[idx] += 1
    return freq.argsort(descending=True)[:num_codebooks]


def _select_greedy(
    all_mse: torch.Tensor,
    num_codebooks: int,
    coverage_threshold: float,
) -> torch.Tensor:
    num_blocks, num_candidates = all_mse.shape
    device = all_mse.device

    optimal_mse = all_mse.min(dim=-1).values
    threshold = optimal_mse * coverage_threshold

    selected: list[int] = []
    covered = torch.zeros(num_blocks, dtype=torch.bool, device=device)

    for _ in range(num_codebooks):
        if covered.all():
            break

        uncovered_mask = ~covered
        uncovered_mse = all_mse[uncovered_mask]
        uncovered_thresh = threshold[uncovered_mask]

        coverage_count = (uncovered_mse <= uncovered_thresh.unsqueeze(1)).sum(dim=0)
        best_candidate = coverage_count.argmax().item()

        while best_candidate in selected:
            coverage_count[best_candidate] = -1
            best_candidate = coverage_count.argmax().item()
            if coverage_count[best_candidate].item() <= 0:
                break

        selected.append(best_candidate)
        newly_covered = all_mse[:, best_candidate] <= threshold
        covered |= newly_covered

    if len(selected) < num_codebooks:
        freq = torch.zeros(num_candidates, dtype=torch.long, device=device)
        winners = all_mse.argmin(dim=-1)
        for idx in winners.cpu().tolist():
            freq[idx] += 1
        selected_set = set(selected)
        remaining = freq.argsort(descending=True)
        for idx in remaining.tolist():
            if len(selected) >= num_codebooks:
                break
            if idx not in selected_set:
                selected.append(idx)
                selected_set.add(idx)

    return torch.tensor(selected[:num_codebooks], dtype=torch.long)


def _sanitize_layer_name(layer_name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]", "_", layer_name)


def _compute_coverage_at_k(
    freq: torch.Tensor, total_blocks: int, k_values: list[int]
) -> dict[str, float]:
    sorted_freq = freq.sort(descending=True).values
    cumsum = sorted_freq.cumsum(dim=0).float()
    total_f = float(total_blocks)
    result = {}
    for k in k_values:
        if k <= len(sorted_freq):
            result[str(k)] = round((cumsum[k - 1] / total_f).item(), 4)
    return result


def _group_layers_by_block(target_layers: list[str]) -> dict[int, list[str]]:
    layers_by_block: dict[int, list[str]] = {}
    for layer_name in target_layers:
        block_idx = layer_block_index(layer_name)
        if block_idx is None:
            raise ValueError(
                f"Target layer is not inside a transformer block: {layer_name}"
            )
        layers_by_block.setdefault(block_idx, []).append(layer_name)
    return layers_by_block


def _layer_complete(output_dir: Path, layer_name: str) -> bool:
    sanitized = _sanitize_layer_name(layer_name)
    return (output_dir / f"{sanitized}.pt").exists() and (
        output_dir / f"{sanitized}.stats.json"
    ).exists()


def _load_block_tensors(
    layer_names: list[str],
    model_path: Path,
    weight_map: dict[str, str],
    input_format: str,
) -> dict[str, torch.Tensor]:
    shard_files: set[str] = set()
    for layer_name in layer_names:
        weight_name = _resolve_weight_name(layer_name, weight_map)
        shard_files.add(weight_map[weight_name])
        if input_format == "nvfp4":
            shard_files.add(weight_map[f"{layer_name}.weight_scale"])
            shard_files.add(weight_map[_resolve_gscale_name(layer_name, weight_map)])

    all_tensors: dict[str, torch.Tensor] = {}
    for sf in shard_files:
        all_tensors.update(load_file(str(model_path / sf), device="cpu"))
    return all_tensors


def _load_hessian_block(hessian_dir: Path, block_idx: int) -> dict[str, torch.Tensor]:
    path = hessian_block_file(hessian_dir, block_idx)
    if not path.exists():
        return {}
    result: dict[str, torch.Tensor] = {}
    with safe_open(str(path), framework="pt", device="cpu") as f:
        for key in f.keys():
            result[key] = f.get_tensor(key)
    return result


def _extract_weight(
    layer_name: str,
    tensors: dict[str, torch.Tensor],
    weight_map: dict[str, str],
    input_format: str,
    device: torch.device,
) -> torch.Tensor:
    weight_name = _resolve_weight_name(layer_name, weight_map)

    if input_format == "bf16":
        return tensors[weight_name].to(device=device, dtype=torch.float32)

    quantizer = CodebookQuantizer()
    scale_name = f"{layer_name}.weight_scale"
    gscale_name = _resolve_gscale_name(layer_name, weight_map)
    packed = tensors[weight_name].to(device=device)
    scale = tensors[scale_name].to(device=device)
    gscale = tensors[gscale_name].to(device=device, dtype=torch.float32).reshape(1)
    fp4_values = quantizer.unpack_uint8_to_fp4(packed)
    scale_expanded = scale.to(torch.float32).repeat_interleave(16, dim=1)
    return fp4_values * scale_expanded * gscale


def _save_layer_result(
    layer_name: str,
    layer_codebook: torch.Tensor,
    num_blocks_layer: int,
    freq: torch.Tensor,
    selected_indices: torch.Tensor,
    num_candidates: int,
    output_dir: Path,
) -> None:
    sanitized = _sanitize_layer_name(layer_name)
    torch.save(layer_codebook, str(output_dir / f"{sanitized}.pt"))

    selected_set = set(selected_indices.tolist())
    selected_freq = sum(freq[k].item() for k in selected_set)
    coverage_256 = selected_freq / num_blocks_layer if num_blocks_layer > 0 else 0.0

    with_zero = sum(
        1 for j in range(layer_codebook.shape[0]) if 0.0 in layer_codebook[j].tolist()
    )
    without_zero = layer_codebook.shape[0] - with_zero
    without_zero_pct = (
        without_zero / layer_codebook.shape[0] if layer_codebook.shape[0] > 0 else 0.0
    )

    coverage_curve = _compute_coverage_at_k(
        freq, num_blocks_layer, [32, 64, 128, 256, 512]
    )

    stats = {
        "num_blocks": num_blocks_layer,
        "num_codebooks": int(layer_codebook.shape[0]),
        "coverage_at_256": round(coverage_256, 4),
        "with_zero_count": with_zero,
        "without_zero_count": without_zero,
        "without_zero_pct": round(without_zero_pct, 4),
        "coverage_curve": coverage_curve,
        "top5": layer_codebook[:5].tolist(),
    }

    with (output_dir / f"{sanitized}.stats.json").open("w") as f:
        json.dump(stats, f)


def _process_block_on_gpu(
    gpu_id: int,
    block_idx: int,
    layer_names: list[str],
    layer_index_map: dict[str, int],
    total_layers: int,
    model_path: Path,
    weight_map: dict[str, str],
    input_format: str,
    hessian_dir: Path,
    output_dir: Path,
    all_codebooks: torch.Tensor,
    num_codebooks: int,
    selection_method: str,
    coverage_threshold: float,
    chunk_size: int,
    continue_existing: bool,
) -> None:
    device = torch.device(f"cuda:{gpu_id}")
    num_candidates = all_codebooks.shape[0]

    pending = [
        n
        for n in layer_names
        if not (continue_existing and _layer_complete(output_dir, n))
    ]
    if not pending:
        skipped = len(layer_names)
        print(
            f"  [GPU {gpu_id}] block {block_idx}: {skipped} layers already complete, skipping"
        )
        return

    block_start = time.perf_counter()

    shard_tensors = _load_block_tensors(pending, model_path, weight_map, input_format)
    hessian_data = _load_hessian_block(hessian_dir, block_idx)

    layer_data: list[tuple[str, int, torch.Tensor, torch.Tensor]] = []
    for layer_name in pending:
        layer_idx = layer_index_map[layer_name]
        w = _extract_weight(layer_name, shard_tensors, weight_map, input_format, device)
        out_features, in_features = w.shape
        if in_features % 16 != 0:
            continue

        H = hessian_data.get(layer_name)
        if H is None:
            print(
                f"  [GPU {gpu_id}] [{layer_idx + 1}/{total_layers}] SKIP (no Hessian): {layer_name}"
            )
            continue

        h_diag = H.diag().to(device) if H.dim() == 2 else H.to(device)
        blocks = w.reshape(-1, 16)
        num_col_groups = in_features // 16
        importance = h_diag.reshape(num_col_groups, 16).repeat(out_features, 1)
        layer_data.append((layer_name, layer_idx, blocks, importance))
        del w, H, h_diag

    del shard_tensors, hessian_data

    if not layer_data:
        return

    offsets: list[tuple[int, int]] = []
    offset = 0
    for _, _, blocks, _ in layer_data:
        n = blocks.shape[0]
        offsets.append((offset, offset + n))
        offset += n

    all_blocks = torch.cat([b for _, _, b, _ in layer_data], dim=0)
    all_importance = torch.cat([imp for _, _, _, imp in layer_data], dim=0)
    total_blocks = all_blocks.shape[0]

    all_winners_list: list[torch.Tensor] = []
    all_mse_list: list[torch.Tensor] = []

    for b_start in range(0, total_blocks, chunk_size):
        b_end = min(b_start + chunk_size, total_blocks)
        winners, mse_matrix = _evaluate_codebooks_batch(
            all_blocks[b_start:b_end], all_importance[b_start:b_end], all_codebooks
        )
        all_winners_list.append(winners.cpu())
        if selection_method == "greedy":
            all_mse_list.append(mse_matrix.cpu())

    del all_blocks, all_importance
    torch.cuda.empty_cache()

    all_winners = torch.cat(all_winners_list)
    all_mse_full = torch.cat(all_mse_list, dim=0) if all_mse_list else None
    del all_winners_list, all_mse_list

    for i, (layer_name, layer_idx, blocks_i, _) in enumerate(layer_data):
        start, end = offsets[i]
        num_blocks_layer = end - start
        layer_winners = all_winners[start:end]

        freq = torch.zeros(num_candidates, dtype=torch.long)
        for idx_val in layer_winners.tolist():
            freq[idx_val] += 1

        if selection_method == "greedy" and all_mse_full is not None:
            selected_indices = _select_greedy(
                all_mse_full[start:end], num_codebooks, coverage_threshold
            )
        else:
            selected_indices = _select_frequency(
                layer_winners, num_codebooks, num_candidates
            )

        layer_codebook = all_codebooks[selected_indices].cpu()
        _save_layer_result(
            layer_name,
            layer_codebook,
            num_blocks_layer,
            freq,
            selected_indices,
            num_candidates,
            output_dir,
        )

    del all_mse_full

    block_elapsed = time.perf_counter() - block_start
    print(
        f"  [GPU {gpu_id}] block {block_idx}: "
        f"{len(layer_data)} layers, {total_blocks} total blocks, {block_elapsed:.1f}s"
    )


def _gpu_worker(
    gpu_id: int,
    assigned_blocks: list[int],
    layers_by_block: dict[int, list[str]],
    layer_index_map: dict[str, int],
    total_layers: int,
    model_path: Path,
    weight_map: dict[str, str],
    input_format: str,
    hessian_dir: Path,
    output_dir: Path,
    num_codebooks: int,
    selection_method: str,
    coverage_threshold: float,
    chunk_size: int,
    continue_existing: bool,
) -> None:
    torch.cuda.set_device(gpu_id)
    all_codebooks = _build_all_candidate_codebooks().to(torch.device(f"cuda:{gpu_id}"))

    for block_idx in assigned_blocks:
        _process_block_on_gpu(
            gpu_id=gpu_id,
            block_idx=block_idx,
            layer_names=layers_by_block[block_idx],
            layer_index_map=layer_index_map,
            total_layers=total_layers,
            model_path=model_path,
            weight_map=weight_map,
            input_format=input_format,
            hessian_dir=hessian_dir,
            output_dir=output_dir,
            all_codebooks=all_codebooks,
            num_codebooks=num_codebooks,
            selection_method=selection_method,
            coverage_threshold=coverage_threshold,
            chunk_size=chunk_size,
            continue_existing=continue_existing,
        )


def _aggregate_results(
    output_dir: Path,
    target_layers: list[str],
    model_path: Path,
    selection_method: str,
    num_codebooks: int,
    num_candidates: int,
    elapsed: float,
) -> None:
    layer_stats: dict[str, dict] = {}
    global_total_blocks = 0
    global_coverage_accum: dict[str, list[float]] = {}
    global_without_zero_pcts: list[float] = []

    for layer_name in target_layers:
        sanitized = _sanitize_layer_name(layer_name)
        stats_path = output_dir / f"{sanitized}.stats.json"
        if not stats_path.exists():
            continue

        with stats_path.open() as f:
            stats = json.load(f)

        global_total_blocks += stats["num_blocks"]
        global_without_zero_pcts.append(stats["without_zero_pct"])

        for k_str, cov_val in stats["coverage_curve"].items():
            global_coverage_accum.setdefault(k_str, []).append(cov_val)

        layer_stats[layer_name] = {
            "num_blocks": stats["num_blocks"],
            "num_codebooks": stats["num_codebooks"],
            "coverage_at_256": stats["coverage_at_256"],
            "with_zero_count": stats["with_zero_count"],
            "without_zero_count": stats["without_zero_count"],
            "top5": stats["top5"],
        }

    global_avg_coverage: dict[str, float] = {}
    for k_str, vals in global_coverage_accum.items():
        global_avg_coverage[k_str] = round(sum(vals) / len(vals), 4) if vals else 0.0

    avg_without_zero_pct = (
        round(sum(global_without_zero_pcts) / len(global_without_zero_pcts), 4)
        if global_without_zero_pcts
        else 0.0
    )

    summary = {
        "model": str(model_path),
        "selection_method": selection_method,
        "num_codebooks_per_layer": num_codebooks,
        "total_layers": len(layer_stats),
        "total_blocks": global_total_blocks,
        "total_candidate_codebooks": num_candidates,
        "analysis_time_seconds": round(elapsed, 1),
        "layers": layer_stats,
        "global_stats": {
            "avg_coverage_at_256": global_avg_coverage.get("256", 0.0),
            "avg_without_zero_pct": avg_without_zero_pct,
            "coverage_curve": global_avg_coverage,
        },
    }

    summary_path = output_dir / "codebook_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== Summary ===")
    print(f"  Total layers: {len(layer_stats)}")
    print(f"  Total blocks: {global_total_blocks}")
    print(f"  Selection method: {selection_method}")
    print(f"  Time: {elapsed:.1f}s")
    for k, v in global_avg_coverage.items():
        print(f"  Avg coverage top-{k}: {v * 100:.1f}%")
    print(f"  Avg without-zero pct: {avg_without_zero_pct * 100:.1f}%")
    print(f"  Saved per-layer codebooks to {output_dir}/")
    print(f"  Saved summary to {summary_path}")


def run_analysis(
    model_path: Path,
    hessian_dir: Path,
    output_dir: Path,
    mlp_only: bool,
    num_codebooks: int,
    selection_method: str,
    coverage_threshold: float,
    device_str: str,
    chunk_size: int,
    num_gpus: int = 1,
    continue_existing: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    weight_map = _load_index(model_path)
    input_format = detect_input_format(weight_map)

    if input_format == "nvfp4":
        all_layers = _find_quantized_layers(weight_map)
    else:
        all_layers = _find_bf16_layers(weight_map)
    target_layers = _filter_layers(all_layers, mlp_only)

    num_candidates = len(list(itertools.combinations(FP4_ALL_VALUES, 4)))

    print(f"Model: {model_path}")
    print(f"Input format: {input_format}")
    print(f"Target layers: {len(target_layers)}")
    print(f"Hessian dir: {hessian_dir}")
    print(f"Selection method: {selection_method}")
    print(f"Num codebooks per layer: {num_codebooks}")
    print(f"Candidate codebooks: {num_candidates}")
    if continue_existing:
        print("Mode: continue (skip already-saved layers)")

    start_time = time.perf_counter()

    layers_by_block = _group_layers_by_block(target_layers)
    sorted_blocks = sorted(layers_by_block.keys())
    effective_gpus = min(
        num_gpus, len(sorted_blocks), max(1, torch.cuda.device_count())
    )

    layer_index_map = {name: idx for idx, name in enumerate(target_layers)}

    gpu_block_assignments: list[list[int]] = [[] for _ in range(effective_gpus)]
    for i, block_idx in enumerate(sorted_blocks):
        gpu_block_assignments[i % effective_gpus].append(block_idx)

    print(f"\n{effective_gpus} GPU(s), {len(sorted_blocks)} blocks")
    for gpu_id, blocks in enumerate(gpu_block_assignments):
        if blocks:
            total_layers_on_gpu = sum(len(layers_by_block[b]) for b in blocks)
            print(f"  GPU {gpu_id}: {len(blocks)} blocks, {total_layers_on_gpu} layers")

    if effective_gpus <= 1:
        gpu_id = 0
        if device_str.startswith("cuda:"):
            gpu_id = int(device_str.split(":")[1])
        _gpu_worker(
            gpu_id=gpu_id,
            assigned_blocks=gpu_block_assignments[0],
            layers_by_block=layers_by_block,
            layer_index_map=layer_index_map,
            total_layers=len(target_layers),
            model_path=model_path,
            weight_map=weight_map,
            input_format=input_format,
            hessian_dir=hessian_dir,
            output_dir=output_dir,
            num_codebooks=num_codebooks,
            selection_method=selection_method,
            coverage_threshold=coverage_threshold,
            chunk_size=chunk_size,
            continue_existing=continue_existing,
        )
    else:
        ctx = mp.get_context("spawn")
        processes: list[mp.Process] = []
        for gpu_id in range(effective_gpus):
            if not gpu_block_assignments[gpu_id]:
                continue
            p = ctx.Process(
                target=_gpu_worker,
                args=(
                    gpu_id,
                    gpu_block_assignments[gpu_id],
                    layers_by_block,
                    layer_index_map,
                    len(target_layers),
                    model_path,
                    weight_map,
                    input_format,
                    hessian_dir,
                    output_dir,
                    num_codebooks,
                    selection_method,
                    coverage_threshold,
                    chunk_size,
                    continue_existing,
                ),
            )
            processes.append(p)
            p.start()

        for p in processes:
            p.join()
            if p.exitcode != 0:
                raise RuntimeError(
                    f"GPU worker {p.name} failed with exit code {p.exitcode}"
                )

    elapsed = time.perf_counter() - start_time
    _aggregate_results(
        output_dir=output_dir,
        target_layers=target_layers,
        model_path=model_path,
        selection_method=selection_method,
        num_codebooks=num_codebooks,
        num_candidates=num_candidates,
        elapsed=elapsed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-layer data-driven codebook analysis with Hessian importance weighting."
    )
    parser.add_argument(
        "--model-path", type=str, default="/data/models/Qwen3-30B-A3B-NVFP4"
    )
    parser.add_argument("--hessian-dir", type=str, default="/data/hessians")
    parser.add_argument("--output-dir", type=str, default="/data/codebooks")
    parser.add_argument("--mlp-only", action="store_true")
    parser.add_argument("--num-codebooks", type=int, default=256)
    parser.add_argument(
        "--selection-method",
        type=str,
        default="greedy",
        choices=["frequency", "greedy"],
    )
    parser.add_argument("--coverage-threshold", type=float, default=1.05)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--chunk-size", type=int, default=16384)
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Number of GPUs for parallel block processing (default: 1)",
    )
    parser.add_argument(
        "--continue",
        dest="continue_existing",
        action="store_true",
        help="Skip layers whose codebook artifacts already exist",
    )
    args = parser.parse_args()

    run_analysis(
        model_path=Path(args.model_path),
        hessian_dir=Path(args.hessian_dir),
        output_dir=Path(args.output_dir),
        mlp_only=args.mlp_only,
        num_codebooks=args.num_codebooks,
        selection_method=args.selection_method,
        coverage_threshold=args.coverage_threshold,
        device_str=args.device,
        chunk_size=args.chunk_size,
        num_gpus=args.num_gpus,
        continue_existing=args.continue_existing,
    )


if __name__ == "__main__":
    main()
