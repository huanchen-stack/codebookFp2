import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from fakequant import CodebookQuantizer
from gptq import CodebookGPTQ

torch.manual_seed(42)

OUT, IN = 256, 512
NUM_SAMPLES = 1024

W = torch.randn(OUT, IN) * 0.3
X = torch.randn(NUM_SAMPLES, IN) * 0.5

q = CodebookQuantizer()

print(f"=== GPTQ test: linear layer ({OUT}x{IN}), {NUM_SAMPLES} calibration samples ===")
print(f"  W range: [{W.min().item():.4f}, {W.max().item():.4f}]")
print(f"  X range: [{X.min().item():.4f}, {X.max().item():.4f}]")

Y_ref = X @ W.T

print("\n--- Plain codebook (no GPTQ) ---")
blocks_plain = W.reshape(-1, 16)
fp4_plain, scale_plain = q.fakequant_blocks_with_scale(blocks_plain)
W_q_plain = (fp4_plain * scale_plain).reshape(OUT, IN)
Y_plain = X @ W_q_plain.T

raw_mse_plain = ((W - W_q_plain) ** 2).mean().item()
output_mse_plain = ((Y_ref - Y_plain) ** 2).mean().item()
print(f"  weight MSE:  {raw_mse_plain:.6f}")
print(f"  output MSE:  {output_mse_plain:.6f}")

print("\n--- GPTQ codebook ---")
gptq = CodebookGPTQ(in_features=IN, quantizer=q)
gptq.update(X)
fp4_gptq, scale_gptq, W_q_gptq = gptq.quantize(W)
Y_gptq = X @ W_q_gptq.T

raw_mse_gptq = ((W - W_q_gptq) ** 2).mean().item()
output_mse_gptq = ((Y_ref - Y_gptq) ** 2).mean().item()
print(f"  weight MSE:  {raw_mse_gptq:.6f}")
print(f"  output MSE:  {output_mse_gptq:.6f}")

print("\n--- Comparison ---")
weight_ratio = raw_mse_gptq / raw_mse_plain
output_ratio = output_mse_gptq / output_mse_plain
print(f"  weight MSE ratio (GPTQ/plain): {weight_ratio:.4f}")
print(f"  output MSE ratio (GPTQ/plain): {output_ratio:.4f}")
print(f"  output MSE reduction:          {(1 - output_ratio) * 100:.1f}%")

blocks_gptq = fp4_gptq.reshape(-1, 16)
max_distinct = max(len(set(blocks_gptq[i].tolist())) for i in range(blocks_gptq.shape[0]))
print(f"  max distinct per block:        {max_distinct}")

assert max_distinct <= 4, f"GPTQ produced block with {max_distinct} distinct values"
assert output_mse_gptq < output_mse_plain, \
    f"GPTQ output MSE ({output_mse_gptq:.6f}) should be < plain ({output_mse_plain:.6f})"
assert not W_q_gptq.isnan().any(), "NaN in GPTQ output"

print("\n--- Hessian-weighted MSE ---")
H = X.T @ X / NUM_SAMPLES
delta_plain = W - W_q_plain
delta_gptq = W - W_q_gptq
h_mse_plain = (delta_plain @ H @ delta_plain.T).trace().item() / OUT
h_mse_gptq = (delta_gptq @ H @ delta_gptq.T).trace().item() / OUT
print(f"  plain:  {h_mse_plain:.6f}")
print(f"  GPTQ:   {h_mse_gptq:.6f}")
print(f"  ratio:  {h_mse_gptq / h_mse_plain:.4f}")

assert h_mse_gptq < h_mse_plain, \
    f"GPTQ Hessian MSE ({h_mse_gptq:.6f}) should be < plain ({h_mse_plain:.6f})"

print("\nAll GPTQ tests passed.")
