from __future__ import annotations

# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnusedCallResult=false

import argparse
import math
import time
from pathlib import Path

import torch
from datasets import load_dataset
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from fakequant import CodebookQuantizer
from fakequant_model import (
    _copy_non_safetensors_files,
    _default_device,
    _filter_layers,
    _find_quantized_layers,
    _load_index,
    _load_tensor_from_specific_shard,
    _resolve_gscale_name,
    _resolve_weight_name,
)
from gptq import CodebookGPTQ


def _collect_hessians(
    calibration_model: str,
    num_samples: int,
    seq_len: int,
    collect_weights: bool = False,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    print(f"\n=== Calibration: {calibration_model} ===")
    tokenizer = AutoTokenizer.from_pretrained(calibration_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        calibration_model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(dataset["text"])
    tokens = tokenizer(text, return_tensors="pt").input_ids[0]

    samples = []
    for i in range(0, len(tokens) - seq_len, seq_len):
        if len(samples) >= num_samples:
            break
        samples.append(tokens[i : i + seq_len].unsqueeze(0))
    print(f"  {len(samples)} sequences x {seq_len} tokens")

    block_layers: dict[int, list[tuple[str, torch.nn.Module]]] = {}
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        parts = name.split(".")
        block_idx = None
        for j, p in enumerate(parts):
            if p == "layers" and j + 1 < len(parts) and parts[j + 1].isdigit():
                block_idx = int(parts[j + 1])
                break
        if block_idx is not None:
            block_layers.setdefault(block_idx, []).append((name, module))

    num_blocks = len(block_layers)
    print(f"  {num_blocks} transformer blocks, processing one at a time")

    hessians: dict[str, torch.Tensor] = {}
    device = next(model.parameters()).device

    for block_idx in sorted(block_layers.keys()):
        layers_in_block = block_layers[block_idx]
        block_hessians: dict[str, torch.Tensor] = {}
        block_counts: dict[str, int] = {}

        def make_hook(layer_name: str):
            def hook_fn(module, inp, out):
                X = inp[0].detach()
                if X.dim() == 3:
                    X = X.reshape(-1, X.shape[-1])
                X = X.float()
                n = X.shape[0]
                d = X.shape[1]
                if layer_name not in block_hessians:
                    block_hessians[layer_name] = torch.zeros(d, d, device=X.device, dtype=torch.float32)
                    block_counts[layer_name] = 0
                sc = block_counts[layer_name]
                beta = sc / (sc + n)
                alpha = 2.0 / (sc + n)
                X_scaled = X.mul(math.sqrt(alpha))
                block_hessians[layer_name].mul_(beta).addmm_(X_scaled.T, X_scaled)
                block_counts[layer_name] = sc + n
            return hook_fn

        hooks = []
        for name, module in layers_in_block:
            hooks.append(module.register_forward_hook(make_hook(name)))

        for i, sample in enumerate(samples):
            with torch.no_grad():
                model(sample.to(device))

        for h in hooks:
            h.remove()

        for name, H in block_hessians.items():
            hessians[name] = H.cpu()
        del block_hessians
        torch.cuda.empty_cache()

        print(f"  block {block_idx}/{num_blocks - 1}: {len(layers_in_block)} layers done")

    bf16_weights: dict[str, torch.Tensor] = {}
    if collect_weights:
        print(f"  Extracting BF16 reference weights...")
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                bf16_weights[name] = module.weight.data.float().cpu()
        print(f"  Extracted {len(bf16_weights)} weight matrices")

    del model
    torch.cuda.empty_cache()

    print(f"  Collected {len(hessians)} Hessians total")

    return hessians, bf16_weights


def _process_shards_gptq(
    input_path: Path,
    output_path: Path,
    weight_map: dict[str, str],
    target_layers: list[str],
    quantizer: CodebookQuantizer,
    device: torch.device,
    hessians: dict[str, torch.Tensor],
    bf16_ref_weights: dict[str, torch.Tensor],
    dry_run: bool,
) -> None:
    shard_order: list[str] = []
    seen: set[str] = set()
    for shard in weight_map.values():
        if shard not in seen:
            seen.add(shard)
            shard_order.append(shard)

    layer_to_idx = {b: idx + 1 for idx, b in enumerate(target_layers)}

    print(f"\nTarget layers: {len(target_layers)} across {len(shard_order)} shard(s).")
    if not dry_run:
        print(f"Using device: {device}")
        print(f"Hessians available: {len(hessians)}")

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
                h_status = "✓" if base in hessians else "✗"
                print(f"  [{idx}/{len(target_layers)}] {base}  shape={tuple(t.shape)}  hessian={h_status}")
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

            packed = packed_cpu.to(device=device)
            scale = scale_cpu.to(device=device)
            gscale = gscale_cpu.to(device=device, dtype=torch.float32).reshape(1)

            fp4_values = quantizer.unpack_uint8_to_fp4(packed)
            out_features, in_features = fp4_values.shape
            scale_expanded = scale.to(torch.float32).repeat_interleave(16, dim=1)

            W_ref = bf16_ref_weights.get(base)
            if W_ref is not None:
                bf16_weights = W_ref.to(device)
                ref_mode = "bf16"
            else:
                bf16_weights = fp4_values * scale_expanded * gscale
                ref_mode = "nvfp4"

            gptq = CodebookGPTQ(in_features=in_features, quantizer=quantizer)

            H = hessians.get(base)
            if H is not None:
                gptq.H = H.to(device)
                gptq.num_samples = 1
            else:
                print(f"  [{idx}/{len(target_layers)}] WARNING: no Hessian for {base}, using random")
                gptq.update(torch.randn(256, in_features, device=device))

            print(f"  [{idx}/{len(target_layers)}] gptq({ref_mode}) {base}")

            fp4_out, scales_out, _ = gptq.quantize(bf16_weights)

            new_packed = quantizer.pack_fp4_to_uint8(fp4_out)
            new_weight_scale = quantizer._cast_scale_to_fp8(scales_out / gscale)

            tensors[weight_name] = new_packed.to(device="cpu")
            tensors[scale_name] = new_weight_scale.to(dtype=scale_cpu.dtype, device="cpu")

            del gptq, bf16_weights, fp4_values
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
    calibration_model: str,
    num_samples: int,
    seq_len: int,
    ref: str = "nvfp4",
    hessian_dir: str | None = None,
) -> None:
    if not input_path.exists() or not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist or is not a directory: {input_path}")

    weight_map = _load_index(input_path)
    all_layers = _find_quantized_layers(weight_map)
    target_layers = _filter_layers(all_layers, mlp_only)

    print(f"Quantized layers in model: {len(all_layers)}")
    print(f"Selected for processing: {len(target_layers)}")
    if mlp_only:
        print(f"Skipped (non-MLP): {len(all_layers) - len(target_layers)}")

    use_bf16_ref = ref == "bf16"

    if hessian_dir is not None:
        hessian_path = Path(hessian_dir)
        print(f"\nLoading pre-computed Hessians from {hessian_path}")
        hessians: dict[str, torch.Tensor] = {}
        for base in target_layers:
            h_file = hessian_path / f"{base}.pt"
            if h_file.exists():
                hessians[base] = torch.load(h_file, map_location="cpu", weights_only=True)
        print(f"  Loaded {len(hessians)}/{len(target_layers)} Hessians")
        bf16_ref_weights: dict[str, torch.Tensor] = {}
        if use_bf16_ref:
            print("WARNING: --ref=bf16 with --hessian-dir requires separate BF16 weight extraction. Falling back to nvfp4.")
            use_bf16_ref = False
    else:
        hessians, bf16_ref_weights = _collect_hessians(
            calibration_model, num_samples, seq_len,
            collect_weights=use_bf16_ref,
        )

    if not use_bf16_ref:
        bf16_ref_weights = {}

    if dry_run:
        print("\n=== DRY RUN (no writes) ===\n")
        _process_shards_gptq(
            input_path=input_path,
            output_path=output_path,
            weight_map=weight_map,
            target_layers=target_layers,
            quantizer=CodebookQuantizer(),
            device=torch.device("cpu"),
            hessians=hessians,
            bf16_ref_weights=bf16_ref_weights,
            dry_run=True,
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
        hessians=hessians,
        bf16_ref_weights=bf16_ref_weights,
        dry_run=False,
    )

    elapsed = time.perf_counter() - start_time
    print(f"Done. Total elapsed time: {elapsed:.2f}s")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply GPTQ-optimized codebook quantization to NVFP4 layers."
    )
    default_input = "models/Qwen3-30B-A3B-NVFP4"
    default_output = default_input + "-CBINT2-GPTQ"
    parser.add_argument("--input-path", type=str, default=default_input)
    parser.add_argument("--output-path", type=str, default=default_output)
    parser.add_argument("--device", type=str, default=_default_device())
    parser.add_argument("--mlp-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--calibration-model", type=str, default="Qwen/Qwen3-30B-A3B",
                        help="HuggingFace model for calibration (default: Qwen/Qwen3-30B-A3B)")
    parser.add_argument("--num-samples", type=int, default=128,
                        help="Calibration sequences (default: 128)")
    parser.add_argument("--seq-len", type=int, default=2048,
                        help="Sequence length per sample (default: 2048)")
    parser.add_argument("--ref", type=str, default="nvfp4", choices=["nvfp4", "bf16"],
                        help="Reference weights: nvfp4=dequantized input (default), bf16=from calibration model")
    parser.add_argument("--hessian-dir", type=str, default=None,
                        help="Load pre-computed Hessians instead of running calibration")
    args = parser.parse_args()

    run(
        input_path=Path(args.input_path),
        output_path=Path(args.output_path),
        device=args.device,
        mlp_only=args.mlp_only,
        dry_run=args.dry_run,
        calibration_model=args.calibration_model,
        num_samples=args.num_samples,
        seq_len=args.seq_len,
        ref=args.ref,
        hessian_dir=args.hessian_dir,
    )


if __name__ == "__main__":
    main()
