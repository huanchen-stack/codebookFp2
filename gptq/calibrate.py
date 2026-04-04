from __future__ import annotations

# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnusedCallResult=false

import argparse
import math
from pathlib import Path

import torch
from datasets import load_dataset
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL_PATH = "Qwen/Qwen3-30B-A3B"
DEFAULT_DATASET_NAME = "wikitext"
DEFAULT_DATASET_CONFIG = "wikitext-2-raw-v1"
MLP_SUFFIXES = (".mlp.gate_proj", ".mlp.up_proj", ".mlp.down_proj")
EXPERT_MLP_PATTERN = ".mlp.experts."
DEFAULT_CAL_BATCH_SIZE = 128


def tokenize_wikitext(
    tokenizer, num_samples: int = 128, seq_len: int = 2048
) -> list[torch.Tensor]:
    dataset = load_dataset(DEFAULT_DATASET_NAME, DEFAULT_DATASET_CONFIG, split="train")
    text = "\n\n".join(dataset["text"])
    tokens = tokenizer(text, return_tensors="pt").input_ids[0]

    samples = []
    for i in range(0, len(tokens) - seq_len, seq_len):
        if len(samples) >= num_samples:
            break
        samples.append(tokens[i : i + seq_len].unsqueeze(0))

    print(
        f"Calibration: {len(samples)} sequences x {seq_len} tokens "
        + f"from {DEFAULT_DATASET_NAME}/{DEFAULT_DATASET_CONFIG}"
    )
    return samples


def _is_mlp_layer(layer_name: str) -> bool:
    return (
        any(layer_name.endswith(suffix) for suffix in MLP_SUFFIXES)
        or EXPERT_MLP_PATTERN in layer_name
    )


def layer_block_index(layer_name: str) -> int | None:
    parts = layer_name.split(".")
    for idx, part in enumerate(parts):
        if part == "layers" and idx + 1 < len(parts) and parts[idx + 1].isdigit():
            return int(parts[idx + 1])
    return None


def hessian_block_file(output_dir: Path, block_idx: int) -> Path:
    return output_dir / f"block_{block_idx:02d}.safetensors"


def hessian_file(output_dir: Path, layer_name: str) -> Path:
    block_idx = layer_block_index(layer_name)
    if block_idx is None:
        raise ValueError(f"Layer is not inside a transformer block: {layer_name}")
    return hessian_block_file(output_dir, block_idx)


def hessian_block_keys(output_dir: Path, block_idx: int) -> set[str]:
    path = hessian_block_file(output_dir, block_idx)
    if not path.exists():
        return set()
    with safe_open(str(path), framework="pt", device="cpu") as f:
        return set(f.keys())


def hessian_block_complete(output_dir: Path, block_idx: int) -> bool:
    return hessian_block_file(output_dir, block_idx).exists()


def load_hessian(output_dir: Path, layer_name: str) -> torch.Tensor | None:
    path = hessian_file(output_dir, layer_name)
    if not path.exists():
        return None
    with safe_open(str(path), framework="pt", device="cpu") as f:
        if layer_name not in f.keys():
            return None
        return f.get_tensor(layer_name)


def _atomic_save_hessian_block(tensors: dict[str, torch.Tensor], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    cpu_tensors = {name: tensor.cpu().contiguous() for name, tensor in tensors.items()}
    save_file(cpu_tensors, str(tmp_path))
    _ = tmp_path.replace(path)


def collect_hessians(
    model_path: str,
    output_dir: Path,
    num_samples: int = 128,
    seq_len: int = 2048,
    dtype: str = "float16",
    mlp_only: bool = False,
    continue_existing: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    torch_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype]

    print(f"Loading tokenizer from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    print(f"Loading model from {model_path} (dtype={dtype}, device_map=auto)")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    # Neuter lm_head: we only need intermediate activations captured by hooks,
    # not the (batch × seq × vocab_size) logits that OOM on gather-back-to-GPU-0.
    # Model is deleted after calibration, so no restore needed.
    with torch.no_grad():
        model.lm_head.weight.data = model.lm_head.weight.data[:1]

    samples = tokenize_wikitext(tokenizer, num_samples=num_samples, seq_len=seq_len)

    block_layers: dict[int, list[tuple[str, torch.nn.Linear]]] = {}
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if mlp_only and not _is_mlp_layer(name):
            continue
        block_idx = layer_block_index(name)
        if block_idx is None:
            continue
        block_layers.setdefault(block_idx, []).append((name, module))

    num_blocks = len(block_layers)
    total_layers = sum(len(layers) for layers in block_layers.values())
    print(
        f"Hook candidates: {total_layers} linear layers across {num_blocks} transformer blocks"
    )
    if mlp_only:
        print("Mode: MLP-only")
    if continue_existing:
        print("Mode: continue (skip already-saved artifacts)")

    input_device = next(model.parameters()).device
    saved_hessians = 0
    skipped_hessians = 0

    for block_idx in sorted(block_layers):
        layers_in_block = block_layers[block_idx]
        layer_names = [layer_name for layer_name, _ in layers_in_block]

        if continue_existing and hessian_block_complete(output_dir, block_idx):
            skipped_hessians += len(layer_names)
            print(f"  block {block_idx}/{num_blocks - 1}: already complete, skipping")
            continue

        block_hessians: dict[str, torch.Tensor] = {}
        block_counts: dict[str, int] = {}

        def make_hook(layer_name: str):
            def hook_fn(_module, inp, _out):
                x = inp[0].detach()
                if x.dim() == 3:
                    x = x.reshape(-1, x.shape[-1])
                x = x.float()
                n = x.shape[0]
                d = x.shape[1]

                if layer_name not in block_hessians:
                    block_hessians[layer_name] = torch.zeros(
                        d, d, dtype=torch.float32, device=x.device
                    )
                    block_counts[layer_name] = 0

                sample_count = block_counts[layer_name]
                beta = sample_count / (sample_count + n)
                alpha = 2.0 / (sample_count + n)
                x_scaled = x.mul(math.sqrt(alpha))
                update = x_scaled.T @ x_scaled
                block_hessians[layer_name].mul_(beta).add_(update)
                block_counts[layer_name] = sample_count + n

            return hook_fn

        hooks = [
            module.register_forward_hook(make_hook(layer_name))
            for layer_name, module in layers_in_block
        ]

        if layers_in_block:
            for batch_start in range(0, len(samples), DEFAULT_CAL_BATCH_SIZE):
                batch = torch.cat(
                    samples[batch_start : batch_start + DEFAULT_CAL_BATCH_SIZE], dim=0
                ).to(input_device)
                with torch.no_grad():
                    model(batch, use_cache=False)
                del batch
                if input_device.type == "cuda":
                    torch.cuda.empty_cache()

        for hook in hooks:
            hook.remove()

        block_file = hessian_block_file(output_dir, block_idx)
        _atomic_save_hessian_block(block_hessians, block_file)
        saved_hessians += len(block_hessians)

        print(
            f"  block {block_idx}/{num_blocks - 1}: "
            + f"saved {len(block_hessians)} Hessians to {block_file.name}"
        )

        del block_hessians
        if input_device.type == "cuda":
            torch.cuda.empty_cache()

    del model
    if input_device.type == "cuda":
        torch.cuda.empty_cache()

    print(f"\nDone. Saved {saved_hessians} Hessians to {output_dir}")
    if continue_existing:
        print(f"Skipped existing Hessians: {skipped_hessians}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect per-layer Hessians for GPTQ calibration."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help=f"HuggingFace model path (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/data/hessians",
        help="Directory to save Hessian .pt files",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=128,
        help="Number of calibration sequences (default: 128)",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=2048,
        help="Sequence length per sample (default: 2048)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
    )
    parser.add_argument(
        "--mlp-only",
        action="store_true",
        help="Collect Hessians only for MLP/expert linear layers",
    )
    parser.add_argument(
        "--continue",
        dest="continue_existing",
        action="store_true",
        help="Skip layers whose calibration artifacts already exist",
    )
    args = parser.parse_args()

    collect_hessians(
        model_path=args.model_path,
        output_dir=Path(args.output_dir),
        num_samples=args.num_samples,
        seq_len=args.seq_len,
        dtype=args.dtype,
        mlp_only=args.mlp_only,
        continue_existing=args.continue_existing,
    )


if __name__ == "__main__":
    main()
