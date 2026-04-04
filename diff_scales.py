import json
import sys
from pathlib import Path

from safetensors import safe_open


def main():
    orig_path = (
        Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/data/models/Qwen3-30B-A3B")
    )
    cbint_path = (
        Path(sys.argv[2]) if len(sys.argv) > 2 else Path(str(orig_path) + "-CBINT2")
    )

    with open(orig_path / "model.safetensors.index.json") as f:
        wmap = json.load(f)["weight_map"]

    scale_keys = sorted(k for k in wmap if k.endswith(".weight_scale"))
    total_groups = 0
    changed_groups = 0
    layers_changed = 0

    for sk in scale_keys:
        shard = wmap[sk]
        with safe_open(str(orig_path / shard), framework="pt", device="cpu") as f:
            orig_s = f.get_tensor(sk)
        with safe_open(str(cbint_path / shard), framework="pt", device="cpu") as f:
            new_s = f.get_tensor(sk)

        diff = (orig_s.float() - new_s.float()).abs()
        n_changed = (diff > 0).sum().item()
        n_total = orig_s.numel()
        total_groups += n_total
        changed_groups += n_changed

        if n_changed > 0:
            layers_changed += 1
            pct = 100.0 * n_changed / n_total
            print(
                f"  {sk:60s}  {n_changed:6d}/{n_total:6d} changed ({pct:5.1f}%)  max_diff={diff.max().item():.4f}"
            )

    print(f"\nSummary:")
    print(f"  layers with scale changes: {layers_changed}/{len(scale_keys)}")
    print(
        f"  total groups changed:      {changed_groups}/{total_groups} ({100.0 * changed_groups / total_groups:.1f}%)"
    )


if __name__ == "__main__":
    main()
