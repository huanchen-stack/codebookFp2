from __future__ import annotations

import argparse
import itertools
import json
import re
import time
from pathlib import Path

import torch
from safetensors.torch import load_file

from fakequant import CodebookQuantizer
from fakequant_model import (
    _filter_layers,
    _find_bf16_layers,
    _find_quantized_layers,
    _load_index,
    _load_tensor_from_specific_shard,
    _resolve_gscale_name,
    _resolve_weight_name,
    detect_input_format,
)
from gptq.calibrate import load_hessian


FP4_ALL_VALUES = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
]


def _build_all_candidate_codebooks() -> torch.Tensor:
    combos = list(itertools.combinations(FP4_ALL_VALUES, 4))
    return torch.tensor(combos, dtype=torch.float32)


def _cast_scale_to_fp8(scale: torch.Tensor) -> torch.Tensor:
    return scale.clamp(-448.0, 448.0).to(torch.float8_e4m3fn).to(torch.float32).clamp(min=1e-10)


def _evaluate_codebooks_chunk(
    blocks: torch.Tensor,
    importance: torch.Tensor,
    all_codebooks: torch.Tensor,
    cb_chunk_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_blocks = blocks.shape[0]
    num_codebooks = all_codebooks.shape[0]
    device = blocks.device

    best_mse = torch.full((num_blocks,), float("inf"), device=device)
    best_k = torch.zeros(num_blocks, dtype=torch.long, device=device)
    all_mse = torch.full((num_blocks, num_codebooks), float("inf"), device=device)

    imp = importance / importance.mean(dim=-1, keepdim=True).clamp(min=1e-10)

    for cb_start in range(0, num_codebooks, cb_chunk_size):
        cb_end = min(cb_start + cb_chunk_size, num_codebooks)
        cb_chunk = all_codebooks[cb_start:cb_end]
        K = cb_chunk.shape[0]

        dists = (blocks[:, :, None, None] - cb_chunk[None, None, :, :]).abs()
        nearest_idx = dists.argmin(dim=-1)
        q_vals = cb_chunk[None, None, :, :].expand(num_blocks, 16, K, 4)
        q_mapped = q_vals.gather(3, nearest_idx.unsqueeze(-1)).squeeze(-1)

        imp_3d = imp.unsqueeze(-1)
        numer = (imp_3d * blocks.unsqueeze(-1) * q_mapped).sum(dim=1)
        denom = (imp_3d * q_mapped ** 2).sum(dim=1).clamp(min=1e-10)

        s_all = _cast_scale_to_fp8(numer / denom)

        w_sq = (imp * blocks ** 2).sum(dim=1, keepdim=True)
        mse_chunk = w_sq - 2 * s_all * numer + s_all ** 2 * denom
        mse_chunk[numer <= 0] = float("inf")

        all_mse[:, cb_start:cb_end] = mse_chunk

        chunk_best_mse, chunk_best_idx = mse_chunk.min(dim=-1)
        improved = chunk_best_mse < best_mse
        best_mse[improved] = chunk_best_mse[improved]
        best_k[improved] = chunk_best_idx[improved] + cb_start

    return best_k, all_mse


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


def _load_layer_weights(
    layer_name: str,
    input_path: Path,
    weight_map: dict[str, str],
    input_format: str,
    device: torch.device,
) -> torch.Tensor:
    weight_name = _resolve_weight_name(layer_name, weight_map)
    shard_file = weight_map[weight_name]
    w_cpu = _load_tensor_from_specific_shard(input_path, shard_file, weight_name)

    if input_format == "nvfp4":
        quantizer = CodebookQuantizer()
        scale_name = f"{layer_name}.weight_scale"
        gscale_name = _resolve_gscale_name(layer_name, weight_map)
        scale_cpu = _load_tensor_from_specific_shard(input_path, weight_map[scale_name], scale_name)
        gscale_cpu = _load_tensor_from_specific_shard(input_path, weight_map[gscale_name], gscale_name)
        packed = w_cpu.to(device=device)
        scale = scale_cpu.to(device=device)
        gscale = gscale_cpu.to(device=device, dtype=torch.float32).reshape(1)
        fp4_values = quantizer.unpack_uint8_to_fp4(packed)
        scale_expanded = scale.to(torch.float32).repeat_interleave(16, dim=1)
        return fp4_values * scale_expanded * gscale
    else:
        return w_cpu.to(device=device, dtype=torch.float32)


def _sanitize_layer_name(layer_name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]", "_", layer_name)


def _compute_coverage_at_k(freq: torch.Tensor, total_blocks: int, k_values: list[int]) -> dict[str, float]:
    sorted_freq = freq.sort(descending=True).values
    cumsum = sorted_freq.cumsum(dim=0).float()
    total_f = float(total_blocks)
    result = {}
    for k in k_values:
        if k <= len(sorted_freq):
            result[str(k)] = round((cumsum[k - 1] / total_f).item(), 4)
    return result


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
) -> None:
    device = torch.device(device_str)
    output_dir.mkdir(parents=True, exist_ok=True)

    weight_map = _load_index(model_path)
    input_format = detect_input_format(weight_map)

    if input_format == "nvfp4":
        all_layers = _find_quantized_layers(weight_map)
    else:
        all_layers = _find_bf16_layers(weight_map)
    target_layers = _filter_layers(all_layers, mlp_only)

    print(f"Model: {model_path}")
    print(f"Input format: {input_format}")
    print(f"Target layers: {len(target_layers)}")
    print(f"Hessian dir: {hessian_dir}")
    print(f"Selection method: {selection_method}")
    print(f"Num codebooks per layer: {num_codebooks}")

    all_codebooks = _build_all_candidate_codebooks().to(device)
    num_candidates = all_codebooks.shape[0]
    print(f"Candidate codebooks: {num_candidates}")

    start_time = time.perf_counter()
    layer_stats: dict[str, dict] = {}
    global_total_blocks = 0
    global_coverage_accum: dict[str, list[float]] = {}
    global_without_zero_pcts: list[float] = []

    for layer_idx, layer_name in enumerate(target_layers):
        layer_start = time.perf_counter()

        w = _load_layer_weights(layer_name, model_path, weight_map, input_format, device)
        out_features, in_features = w.shape
        assert in_features % 16 == 0

        H = load_hessian(hessian_dir, layer_name)
        if H is None:
            print(f"  [{layer_idx + 1}/{len(target_layers)}] SKIP (no Hessian): {layer_name}")
            continue

        h_diag = H.diag().to(device) if H.dim() == 2 else H.to(device)

        blocks = w.reshape(-1, 16)
        num_col_groups = in_features // 16
        importance_per_group = h_diag.reshape(num_col_groups, 16)
        importance = importance_per_group.repeat(out_features, 1)

        num_blocks_layer = blocks.shape[0]
        global_total_blocks += num_blocks_layer

        all_winners_list: list[torch.Tensor] = []
        all_mse_list: list[torch.Tensor] = []

        for b_start in range(0, num_blocks_layer, chunk_size):
            b_end = min(b_start + chunk_size, num_blocks_layer)
            chunk_blocks = blocks[b_start:b_end]
            chunk_imp = importance[b_start:b_end]

            winners, mse_matrix = _evaluate_codebooks_chunk(chunk_blocks, chunk_imp, all_codebooks)
            all_winners_list.append(winners.cpu())
            if selection_method == "greedy":
                all_mse_list.append(mse_matrix.cpu())

        all_winners = torch.cat(all_winners_list)

        freq = torch.zeros(num_candidates, dtype=torch.long)
        for idx in all_winners.tolist():
            freq[idx] += 1

        if selection_method == "greedy" and all_mse_list:
            all_mse_full = torch.cat(all_mse_list, dim=0)
            selected_indices = _select_greedy(all_mse_full, num_codebooks, coverage_threshold)
            del all_mse_full
        else:
            selected_indices = _select_frequency(all_winners, num_codebooks, num_candidates)

        layer_codebook = all_codebooks[selected_indices].cpu()

        sanitized = _sanitize_layer_name(layer_name)
        layer_pt = output_dir / f"{sanitized}.pt"
        torch.save(layer_codebook, str(layer_pt))

        selected_set = set(selected_indices.tolist())
        selected_freq = sum(freq[i].item() for i in selected_set)
        coverage_256 = selected_freq / num_blocks_layer if num_blocks_layer > 0 else 0.0

        with_zero = sum(1 for i in range(layer_codebook.shape[0]) if 0.0 in layer_codebook[i].tolist())
        without_zero = layer_codebook.shape[0] - with_zero
        without_zero_pct = without_zero / layer_codebook.shape[0] if layer_codebook.shape[0] > 0 else 0.0
        global_without_zero_pcts.append(without_zero_pct)

        coverage_curve = _compute_coverage_at_k(freq, num_blocks_layer, [32, 64, 128, 256, 512])
        for k_str, cov_val in coverage_curve.items():
            global_coverage_accum.setdefault(k_str, []).append(cov_val)

        top5 = layer_codebook[:5].tolist()

        layer_stats[layer_name] = {
            "num_blocks": num_blocks_layer,
            "num_codebooks": int(layer_codebook.shape[0]),
            "coverage_at_256": round(coverage_256, 4),
            "with_zero_count": with_zero,
            "without_zero_count": without_zero,
            "top5": top5,
        }

        layer_elapsed = time.perf_counter() - layer_start
        print(
            f"  [{layer_idx + 1}/{len(target_layers)}] {layer_name}: "
            f"{num_blocks_layer} blocks, coverage={coverage_256:.1%}, "
            f"zero={with_zero}/no-zero={without_zero}, {layer_elapsed:.1f}s"
        )

        del w, blocks, importance, H, h_diag, all_winners_list, all_mse_list
        if device.type == "cuda":
            torch.cuda.empty_cache()

    elapsed = time.perf_counter() - start_time

    global_avg_coverage: dict[str, float] = {}
    for k_str, vals in global_coverage_accum.items():
        global_avg_coverage[k_str] = round(sum(vals) / len(vals), 4) if vals else 0.0

    avg_without_zero_pct = round(
        sum(global_without_zero_pcts) / len(global_without_zero_pcts), 4
    ) if global_without_zero_pcts else 0.0

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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-layer data-driven codebook analysis with Hessian importance weighting."
    )
    parser.add_argument("--model-path", type=str, default="/data/models/Qwen3-30B-A3B-NVFP4")
    parser.add_argument("--hessian-dir", type=str, default="/data/hessians")
    parser.add_argument("--output-dir", type=str, default="/data/codebooks")
    parser.add_argument("--mlp-only", action="store_true")
    parser.add_argument("--num-codebooks", type=int, default=256)
    parser.add_argument("--selection-method", type=str, default="greedy",
                        choices=["frequency", "greedy"])
    parser.add_argument("--coverage-threshold", type=float, default=1.05)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--chunk-size", type=int, default=4096)
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
    )


if __name__ == "__main__":
    main()
