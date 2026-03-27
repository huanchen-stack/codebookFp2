from fakequant import CodebookQuantizer
import torch

torch.manual_seed(42)
q = CodebookQuantizer()

print("=== Nibble-to-FP4 lookup table ===")
for i in range(16):
    print(f"  nibble {i:2d} (0b{i:04b}) -> {q.nibble_to_fp4[i].item():+.1f}")

print(f"\n=== Codebook: {q.codebook.shape[0]} entries (first 5) ===")
for i in range(5):
    vals = q.codebook[i].tolist()
    print(f"  entry {i}: {[f'{v:+.1f}' for v in vals]}")
print(f"  ...")

print("\n=== Single block fakequant ===")
block = torch.tensor([[0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                        -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0, 0.0]])
print(f"  original:   {block[0].tolist()}")
quantized = q.fakequant_blocks(block)
print(f"  quantized:  {quantized[0].tolist()}")
unique = sorted(set(quantized[0].tolist()))
print(f"  distinct values: {unique} (count: {len(unique)})")
mse = ((block - quantized) ** 2).mean().item()
print(f"  MSE: {mse:.6f}")

print("\n=== Random blocks (8 blocks of 16) ===")
random_blocks = q.fp4_representable[torch.randint(0, 15, (8, 16))]
quantized_random = q.fakequant_blocks(random_blocks)
for i in range(8):
    orig = [f"{v:+.1f}" for v in random_blocks[i].tolist()]
    qval = [f"{v:+.1f}" for v in quantized_random[i].tolist()]
    unique_q = sorted(set(quantized_random[i].tolist()))
    block_mse = ((random_blocks[i] - quantized_random[i]) ** 2).mean().item()
    print(f"  block {i}:")
    print(f"    orig: {orig}")
    print(f"    qnt:  {qval}")
    print(f"    unique: {[f'{v:+.1f}' for v in unique_q]} | MSE: {block_mse:.4f}")

print("\n=== Pack/unpack roundtrip ===")
packed = q.pack_fp4_to_uint8(quantized_random)
unpacked = q.unpack_uint8_to_fp4(packed)
match = torch.equal(unpacked, quantized_random)
print(f"  roundtrip exact match: {match}")

print("\n=== Full layer fakequant (32x512) ===")
layer_fp4 = q.fp4_representable[torch.randint(0, 15, (32, 512))]
layer_packed = q.pack_fp4_to_uint8(layer_fp4)
scale = torch.ones((32, 32), dtype=torch.float32)
gscale = torch.ones((1,), dtype=torch.float32)

quantized_packed = q.fakequant_layer(layer_packed, scale, gscale)
quantized_layer = q.unpack_uint8_to_fp4(quantized_packed)

layer_mse = ((layer_fp4 - quantized_layer) ** 2).mean().item()
blocks_q = quantized_layer.reshape(-1, 16)
max_distinct = max(len(set(blocks_q[i].tolist())) for i in range(blocks_q.shape[0]))

print(f"  shape: {layer_fp4.shape} -> {quantized_layer.shape}")
print(f"  overall MSE: {layer_mse:.6f}")
print(f"  max distinct values per block: {max_distinct}")
print(f"  first block orig:  {[f'{v:+.1f}' for v in layer_fp4[0, :16].tolist()]}")
print(f"  first block quant: {[f'{v:+.1f}' for v in quantized_layer[0, :16].tolist()]}")
