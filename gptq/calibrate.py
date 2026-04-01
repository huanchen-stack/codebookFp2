from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def tokenize_wikitext(tokenizer, num_samples: int = 128, seq_len: int = 2048) -> list[torch.Tensor]:
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(dataset["text"])
    tokens = tokenizer(text, return_tensors="pt").input_ids[0]

    samples = []
    for i in range(0, len(tokens) - seq_len, seq_len):
        if len(samples) >= num_samples:
            break
        samples.append(tokens[i : i + seq_len].unsqueeze(0))

    print(f"Calibration: {len(samples)} sequences x {seq_len} tokens")
    return samples


def collect_hessians(
    model_path: str,
    output_dir: Path,
    num_samples: int = 128,
    seq_len: int = 2048,
    dtype: str = "float16",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype]

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

    samples = tokenize_wikitext(tokenizer, num_samples=num_samples, seq_len=seq_len)

    hessians: dict[str, torch.Tensor] = {}
    sample_counts: dict[str, int] = {}

    def make_hook(layer_name: str):
        def hook_fn(module, inp, out):
            X = inp[0]
            if X.dim() == 3:
                X = X.reshape(-1, X.shape[-1])
            X = X.float()
            n = X.shape[0]
            d = X.shape[1]

            if layer_name not in hessians:
                hessians[layer_name] = torch.zeros(d, d, device=X.device, dtype=torch.float32)
                sample_counts[layer_name] = 0

            H = hessians[layer_name]
            sc = sample_counts[layer_name]
            beta = sc / (sc + n)
            alpha = 2.0 / (sc + n)
            H.mul_(beta)
            X_scaled = X.mul(math.sqrt(alpha))
            H.addmm_(X_scaled.T, X_scaled)
            sample_counts[layer_name] = sc + n

        return hook_fn

    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            hooks.append(module.register_forward_hook(make_hook(name)))

    print(f"Hooked {len(hooks)} linear layers")
    print(f"Running {len(samples)} calibration forward passes...")

    for i, sample in enumerate(samples):
        device = next(model.parameters()).device
        with torch.no_grad():
            model(sample.to(device))
        if (i + 1) % 16 == 0 or i == len(samples) - 1:
            print(f"  [{i + 1}/{len(samples)}]")

    for h in hooks:
        h.remove()

    print(f"\nSaving {len(hessians)} Hessians to {output_dir}")
    for name, H in hessians.items():
        save_path = output_dir / f"{name}.pt"
        torch.save(H.cpu(), save_path)

    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect per-layer Hessians for GPTQ calibration.")
    parser.add_argument("--model-path", type=str, required=True,
                        help="HuggingFace model path (local or hub)")
    parser.add_argument("--output-dir", type=str, default="hessians",
                        help="Directory to save Hessian .pt files")
    parser.add_argument("--num-samples", type=int, default=128,
                        help="Number of calibration sequences (default: 128)")
    parser.add_argument("--seq-len", type=int, default=2048,
                        help="Sequence length per sample (default: 2048)")
    parser.add_argument("--dtype", type=str, default="float16",
                        choices=["float16", "bfloat16", "float32"])
    args = parser.parse_args()

    collect_hessians(
        model_path=args.model_path,
        output_dir=Path(args.output_dir),
        num_samples=args.num_samples,
        seq_len=args.seq_len,
        dtype=args.dtype,
    )


if __name__ == "__main__":
    main()
