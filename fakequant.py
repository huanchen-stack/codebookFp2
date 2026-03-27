from __future__ import annotations

import itertools
from typing import Final

import torch


class CodebookQuantizer:
    FP4_MAGNITUDES: Final[tuple[float, ...]] = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
    MAX_CODEBOOK_TENSOR_BYTES: Final[int] = 4 * 1024 * 1024 * 1024

    def __init__(self, policy: str = "top3_nonzero"):
        if policy != "top3_nonzero":
            raise NotImplementedError(
                f"Unsupported policy '{policy}'. Only 'top3_nonzero' is implemented."
            )

        self.policy = policy
        self.fp4_magnitudes = torch.tensor(self.FP4_MAGNITUDES, dtype=torch.float32)

        self.fp4_representable = torch.tensor(
            [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
            dtype=torch.float32,
        )
        self.fp4_nonzero_values = self.fp4_representable[1:]

        self.nibble_to_fp4 = self._build_nibble_to_fp4_table()
        self.codebook = self._build_codebook()

    def _build_nibble_to_fp4_table(self):
        nibbles = torch.arange(16, dtype=torch.int64)
        magnitude_idx = nibbles & 0x7
        sign_bit = (nibbles >> 3) & 0x1

        magnitudes = self.fp4_magnitudes[magnitude_idx]
        values = torch.where(sign_bit.bool(), -magnitudes, magnitudes)
        values[8] = 0.0
        return values.to(torch.float32)

    def _build_codebook(self):
        combos = list(itertools.combinations(self.fp4_nonzero_values.tolist(), 3))
        codebook_rows = [[0.0, combo[0], combo[1], combo[2]] for combo in combos]
        codebook = torch.tensor(codebook_rows, dtype=torch.float32)

        expected_shape = (364, 4)
        if tuple(codebook.shape) != expected_shape:
            raise RuntimeError(
                f"Unexpected codebook shape {tuple(codebook.shape)}; expected {expected_shape}."
            )
        return codebook

    def unpack_uint8_to_fp4(self, packed):
        if packed.dtype != torch.uint8:
            raise TypeError(f"packed must be torch.uint8, got {packed.dtype}")

        low_nibble = packed & 0x0F
        high_nibble = (packed >> 4) & 0x0F

        table = self.nibble_to_fp4.to(device=packed.device)
        low_values = table[low_nibble.long()]
        high_values = table[high_nibble.long()]

        out_shape = list(packed.shape)
        out_shape[-1] *= 2
        fp4_values = torch.empty(out_shape, dtype=torch.float32, device=packed.device)
        fp4_values[..., 0::2] = low_values
        fp4_values[..., 1::2] = high_values
        return fp4_values

    def pack_fp4_to_uint8(self, fp4_values):
        if fp4_values.shape[-1] % 2 != 0:
            raise ValueError(f"Last dimension must be even, got {fp4_values.shape[-1]}")

        values = fp4_values.to(dtype=torch.float32)
        magnitudes = self.fp4_magnitudes.to(device=values.device)

        abs_values = values.abs()
        magnitude_diffs = torch.abs(abs_values.unsqueeze(-1) - magnitudes)
        magnitude_idx = magnitude_diffs.argmin(dim=-1).to(torch.uint8)

        nearest_mag = magnitudes[magnitude_idx.long()]
        max_err = torch.max(torch.abs(nearest_mag - abs_values))
        if max_err.item() > 1e-6:
            raise ValueError("fp4_values contains values not representable in FP4 E2M1 set")

        sign_bit = (values < 0).to(torch.uint8)
        nibbles = (sign_bit << 3) | magnitude_idx

        low_nibbles = nibbles[..., 0::2]
        high_nibbles = nibbles[..., 1::2]
        packed = (low_nibbles | (high_nibbles << 4)).to(torch.uint8)
        return packed.contiguous()

    def fakequant_layer(
        self,
        weight_packed,
        weight_scale,
        weight_global_scale,
    ):
        if weight_packed.dtype != torch.uint8:
            raise TypeError(f"weight_packed must be torch.uint8, got {weight_packed.dtype}")
        if weight_packed.dim() != 2:
            raise ValueError(f"weight_packed must be 2D [out, in//2], got {tuple(weight_packed.shape)}")

        out_features, in_packed = weight_packed.shape
        in_features = in_packed * 2
        if in_features % 16 != 0:
            raise ValueError(f"in_features ({in_features}) must be divisible by 16")

        expected_scale_shape = (out_features, in_features // 16)
        if tuple(weight_scale.shape) != expected_scale_shape:
            raise ValueError(
                f"weight_scale shape {tuple(weight_scale.shape)} does not match expected {expected_scale_shape}"
            )
        if weight_global_scale.numel() != 1:
            raise ValueError(
                f"weight_global_scale must contain exactly one element, got shape {tuple(weight_global_scale.shape)}"
            )

        fp4_values = self.unpack_uint8_to_fp4(weight_packed)
        blocks = fp4_values.reshape(-1, 16)
        quantized_blocks = self.fakequant_blocks(blocks)
        quantized_fp4 = quantized_blocks.reshape_as(fp4_values)
        return self.pack_fp4_to_uint8(quantized_fp4)

    def fakequant_blocks(self, fp4_values):
        if fp4_values.dim() != 2 or fp4_values.shape[1] != 16:
            raise ValueError(
                f"fp4_values must have shape [num_blocks, 16], got {tuple(fp4_values.shape)}"
            )

        source = fp4_values.to(dtype=torch.float32)
        num_blocks, block_size = source.shape
        if num_blocks == 0:
            return source.clone()

        full_tensor_bytes = num_blocks * self.codebook.shape[0] * block_size * 4
        max_blocks_per_chunk = max(
            1,
            self.MAX_CODEBOOK_TENSOR_BYTES // (self.codebook.shape[0] * block_size * 4),
        )

        codebook = self.codebook.to(device=source.device, dtype=torch.float32)
        if full_tensor_bytes <= self.MAX_CODEBOOK_TENSOR_BYTES:
            return self._fakequant_blocks_chunk(source, codebook)

        quantized_chunks = []
        for start in range(0, num_blocks, max_blocks_per_chunk):
            end = min(start + max_blocks_per_chunk, num_blocks)
            quantized_chunks.append(self._fakequant_blocks_chunk(source[start:end], codebook))
        return torch.cat(quantized_chunks, dim=0)

    @staticmethod
    def _fakequant_blocks_chunk(source, codebook):
        num_blocks = source.shape[0]
        best_mse = torch.full((num_blocks,), float("inf"), dtype=torch.float32, device=source.device)
        best_quantized = torch.empty_like(source, dtype=torch.float32)
        source_expanded = source.unsqueeze(-1)

        for codebook_idx in range(codebook.shape[0]):
            entry = codebook[codebook_idx]
            sq_error = (source_expanded - entry.view(1, 1, 4)) ** 2
            nearest_idx = sq_error.argmin(dim=-1)
            nearest_sq_error = sq_error.gather(dim=-1, index=nearest_idx.unsqueeze(-1)).squeeze(-1)
            mse = nearest_sq_error.mean(dim=1)

            better = mse < best_mse
            if better.any():
                best_mse = torch.where(better, mse, best_mse)
                quantized = entry[nearest_idx]
                best_quantized[better] = quantized[better]

        return best_quantized


if __name__ == "__main__":
    torch.manual_seed(0)
    quantizer = CodebookQuantizer()

    representable = quantizer.fp4_representable

    random_fp4 = representable[torch.randint(0, representable.numel(), (64, 256))]
    packed = quantizer.pack_fp4_to_uint8(random_fp4)
    unpacked = quantizer.unpack_uint8_to_fp4(packed)
    assert torch.equal(unpacked, random_fp4), "Pack/unpack roundtrip failed"

    blocks = representable[torch.randint(0, representable.numel(), (1024, 16))]
    quantized_blocks = quantizer.fakequant_blocks(blocks)
    present = (quantized_blocks.unsqueeze(-1) == representable.view(1, 1, -1)).any(dim=1)
    distinct_counts = present.sum(dim=1)
    assert int(distinct_counts.max().item()) <= 4, "Block has more than 4 distinct values"

    out_features = 32
    in_features = 512
    layer_fp4 = representable[torch.randint(0, representable.numel(), (out_features, in_features))]
    layer_packed = quantizer.pack_fp4_to_uint8(layer_fp4)
    weight_scale = torch.ones((out_features, in_features // 16), dtype=torch.float32)
    weight_global_scale = torch.ones((1,), dtype=torch.float32)

    quantized_packed = quantizer.fakequant_layer(layer_packed, weight_scale, weight_global_scale)
    quantized_fp4 = quantizer.unpack_uint8_to_fp4(quantized_packed)
    quantized_blocks_from_layer = quantized_fp4.reshape(-1, 16)
    present_layer = (
        quantized_blocks_from_layer.unsqueeze(-1) == representable.view(1, 1, -1)
    ).any(dim=1)
    distinct_counts_layer = present_layer.sum(dim=1)
    assert (
        int(distinct_counts_layer.max().item()) <= 4
    ), "fakequant_layer produced a block with more than 4 distinct values"

    repacked = quantizer.pack_fp4_to_uint8(quantized_fp4)
    re_unpacked = quantizer.unpack_uint8_to_fp4(repacked)
    assert torch.equal(re_unpacked, quantized_fp4), "Unpack/pack/unpack roundtrip failed"

    print("All fakequant.py self-tests passed.")
