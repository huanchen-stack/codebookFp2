from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract per-layer weight_scale_2 (global scale) from an nvfp4 model."
    )
    parser.add_argument(
        "--model-path", type=str, default="/data/models/Qwen3-30B-A3B-NVFP4"
    )
    parser.add_argument("--output", type=str, default="/data/global_scales.safetensors")
    args = parser.parse_args()

    model_path = Path(args.model_path)
    index_path = model_path / "model.safetensors.index.json"
    with index_path.open() as f:
        weight_map = json.load(f)["weight_map"]

    scale_keys = sorted(k for k in weight_map if k.endswith(".weight_scale_2"))
    print(f"Found {len(scale_keys)} global scale tensors")

    scales: dict[str, torch.Tensor] = {}
    loaded_shards: dict[str, dict[str, torch.Tensor]] = {}

    for key in scale_keys:
        shard = weight_map[key]
        if shard not in loaded_shards:
            shard_path = model_path / shard
            with safe_open(str(shard_path), framework="pt", device="cpu") as f:
                loaded_shards[shard] = {
                    name: f.get_tensor(name)
                    for name in f.keys()
                    if name.endswith(".weight_scale_2")
                }
        scales[key] = loaded_shards[shard][key]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    from safetensors.torch import save_file

    save_file(scales, str(output_path))

    print(f"Saved {len(scales)} global scales to {output_path}")
    for k in list(scales.keys())[:5]:
        v = scales[k].to(torch.float32)
        print(f"  {k}: {v.item():.6f}")
    if len(scales) > 5:
        print(f"  ... ({len(scales) - 5} more)")


if __name__ == "__main__":
    main()
