from __future__ import annotations

import itertools
from typing import Final

import torch


class CodebookQuantizer:
    FP4_MAGNITUDES: Final[tuple[float, ...]] = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
    MAX_CODEBOOK_TENSOR_BYTES: Final[int] = 4 * 1024 * 1024 * 1024

    def __init__(self, policy: str = "top3_nonzero", codebook_path: str | None = None):
        self.policy = policy
        self.fp4_magnitudes = torch.tensor(self.FP4_MAGNITUDES, dtype=torch.float32)

        self.fp4_representable = torch.tensor(
            [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
            dtype=torch.float32,
        )
        self.fp4_nonzero_values = self.fp4_representable[1:]

        self.nibble_to_fp4 = self._build_nibble_to_fp4_table()

        if policy == "top3_nonzero":
            self.codebook = self._build_codebook()
        elif policy == "statistical":
            if codebook_path is None:
                raise ValueError("codebook_path required for policy='statistical'")
            self.codebook = torch.load(codebook_path, weights_only=True).to(torch.float32)
            if self.codebook.dim() != 2 or self.codebook.shape[1] != 4:
                raise ValueError(
                    f"Statistical codebook must have shape [K, 4], got {tuple(self.codebook.shape)}"
                )
        else:
            raise NotImplementedError(f"Unsupported policy '{policy}'.")

        self._error_table, self._value_table = self._build_lookup_tables()

    def _build_nibble_to_fp4_table(self):
        nibbles = torch.arange(16, dtype=torch.int64)
        magnitude_idx = nibbles & 0x7
        sign_bit = (nibbles >> 3) & 0x1

        magnitudes = self.fp4_magnitudes[magnitude_idx]
        values = torch.where(sign_bit.bool(), -magnitudes, magnitudes)
        values[8] = 0.0
        return values.to(torch.float32)

    def set_codebook(self, codebook: torch.Tensor) -> None:
        if codebook.dim() != 2 or codebook.shape[1] != 4:
            raise ValueError(f"Codebook must be [K, 4], got {tuple(codebook.shape)}")
        self.codebook = codebook.to(torch.float32)
        self._error_table, self._value_table = self._build_lookup_tables()

    def _build_codebook(self):
        combos = list(itertools.combinations(self.fp4_nonzero_values.tolist(), 3))
        codebook_rows = [[0.0, combo[0], combo[1], combo[2]] for combo in combos]
        codebook = torch.tensor(codebook_rows, dtype=torch.float32)
        assert codebook.shape == (364, 4)
        return codebook

    def _build_lookup_tables(self):
        fp4 = self.fp4_representable
        cb = self.codebook
        sq_errors = (fp4[:, None, None] - cb[None, :, :]) ** 2
        nearest_k = sq_errors.argmin(dim=-1)
        error_table = sq_errors.gather(-1, nearest_k.unsqueeze(-1)).squeeze(-1)
        c_range = torch.arange(cb.shape[0]).unsqueeze(0).expand(fp4.shape[0], -1)
        value_table = cb[c_range, nearest_k]
        return error_table, value_table

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

    def _fakequant_layer_vanilla(
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
        gscale = weight_global_scale.to(torch.float32).reshape(1)
        scale_expanded = weight_scale.to(torch.float32).repeat_interleave(16, dim=1)
        bf16_weights = fp4_values * scale_expanded * gscale

        blocks = bf16_weights.reshape(-1, 16)
        opt_fp4, opt_scale = self.fakequant_blocks_with_scale(blocks)

        opt_fp4 = opt_fp4.reshape(out_features, in_features)
        new_effective_scale = opt_scale.reshape(out_features, in_features // 16)
        new_weight_scale = self._cast_scale_to_fp8(new_effective_scale / gscale)

        return self.pack_fp4_to_uint8(opt_fp4), new_weight_scale.to(dtype=weight_scale.dtype)

    def fakequant_blocks(self, fp4_values, return_mse: bool = False):
        if fp4_values.dim() != 2 or fp4_values.shape[1] != 16:
            raise ValueError(
                f"fp4_values must have shape [num_blocks, 16], got {tuple(fp4_values.shape)}"
            )

        source = fp4_values.to(dtype=torch.float32)
        num_blocks, block_size = source.shape
        if num_blocks == 0:
            empty = source.clone()
            if return_mse:
                return empty, torch.empty((0, self.codebook.shape[0]), dtype=torch.float32)
            return empty

        full_tensor_bytes = num_blocks * self.codebook.shape[0] * block_size * 4
        max_blocks_per_chunk = max(
            1,
            self.MAX_CODEBOOK_TENSOR_BYTES // (self.codebook.shape[0] * block_size * 4),
        )

        fp4 = self.fp4_representable.to(device=source.device)
        error_table = self._error_table.to(device=source.device)
        value_table = self._value_table.to(device=source.device)

        if full_tensor_bytes <= self.MAX_CODEBOOK_TENSOR_BYTES:
            result = self._fakequant_blocks_chunk(source, fp4, error_table, value_table)
        else:
            q_chunks = []
            mse_chunks = []
            for start in range(0, num_blocks, max_blocks_per_chunk):
                end = min(start + max_blocks_per_chunk, num_blocks)
                quantized, mse = self._fakequant_blocks_chunk(
                    source[start:end], fp4, error_table, value_table
                )
                q_chunks.append(quantized)
                mse_chunks.append(mse)
            result = (torch.cat(q_chunks, dim=0), torch.cat(mse_chunks, dim=0))

        if return_mse:
            return result
        if isinstance(result, tuple):
            return result[0]
        return result

    FP8_E4M3_MAX: Final[float] = 448.0

    @staticmethod
    def _cast_scale_to_fp8(scale):
        return scale.clamp(-448.0, 448.0).to(torch.float8_e4m3fn).to(torch.float32).clamp(min=1e-10)

    def _round_to_fp4_indices(self, values):
        fp4 = self.fp4_representable.to(device=values.device)
        return (values.unsqueeze(-1) - fp4).abs().argmin(dim=-1)

    def fakequant_blocks_with_scale(
        self,
        bf16_weights,
        return_codebook_idx: bool = False,
        importance_weights: torch.Tensor | None = None,
    ):
        if bf16_weights.dim() != 2 or bf16_weights.shape[1] != 16:
            raise ValueError(
                f"bf16_weights must have shape [num_blocks, 16], got {tuple(bf16_weights.shape)}"
            )

        w = bf16_weights.to(dtype=torch.float32)
        device = w.device

        fp4 = self.fp4_representable.to(device=device)
        value_table = self._value_table.to(device=device)

        s_init = self._cast_scale_to_fp8(w.abs().amax(dim=-1, keepdim=True) / 6.0)
        fp4_idx = (w / s_init).unsqueeze(-1).sub(fp4).abs().argmin(dim=-1)

        q_all = value_table[fp4_idx]

        if importance_weights is not None:
            imp = importance_weights.to(dtype=torch.float32, device=device)
            imp = imp / imp.mean(dim=-1, keepdim=True).clamp(min=1e-10)
            imp_3d = imp.unsqueeze(-1)
            numer = (imp_3d * w.unsqueeze(-1) * q_all).sum(dim=1)
            denom = (imp_3d * q_all ** 2).sum(dim=1).clamp(min=1e-10)
            s_all = self._cast_scale_to_fp8(numer / denom)
            w_sq = (imp * w ** 2).sum(dim=1, keepdim=True)
            mse_all = w_sq - 2 * s_all * numer + s_all ** 2 * denom
        else:
            numer = (w.unsqueeze(-1) * q_all).sum(dim=1)
            denom = (q_all ** 2).sum(dim=1).clamp(min=1e-10)
            s_all = self._cast_scale_to_fp8(numer / denom)
            w_sq = (w ** 2).sum(dim=1, keepdim=True)
            mse_all = w_sq - 2 * s_all * numer + s_all ** 2 * denom

        mse_all[numer <= 0] = float("inf")

        best_k = mse_all.argmin(dim=-1)

        q = q_all.gather(2, best_k.view(-1, 1, 1).expand(-1, 16, 1)).squeeze(2)
        s = s_all.gather(1, best_k.unsqueeze(1))

        if return_codebook_idx:
            return q, s, best_k
        return q, s

    def fakequant_layer_bf16(self, bf16_weights: torch.Tensor) -> torch.Tensor:
        """Codebook-quantize full-precision weights and return dequantized BF16 result.

        Input:  bf16_weights  — shape [out_features, in_features], any float dtype
        Output: bf16 tensor   — shape [out_features, in_features], codebook-constrained values
                                (= codebook_fp4 * optimized_scale, baked into BF16)

        The returned tensor has the same shape as the input but each 16-element
        block is constrained to at most 4 distinct FP4 values times a per-block
        FP8-rounded scale.  This is the A100-compatible path: no FP4 tensor cores
        required at inference time.
        """
        w = bf16_weights.to(dtype=torch.float32)
        if w.dim() != 2:
            raise ValueError(f"bf16_weights must be 2D [out, in], got {tuple(w.shape)}")
        out_features, in_features = w.shape
        if in_features % 16 != 0:
            raise ValueError(f"in_features ({in_features}) must be divisible by 16")

        blocks = w.reshape(-1, 16)
        opt_fp4, opt_scale = self.fakequant_blocks_with_scale(blocks)
        dequantized = (opt_fp4 * opt_scale).reshape(out_features, in_features)
        return dequantized.to(dtype=bf16_weights.dtype)

    @staticmethod
    def _fakequant_blocks_chunk(source, fp4, error_table, value_table):
        element_indices = (source.unsqueeze(-1) == fp4).int().argmax(dim=-1)
        element_errors = error_table[element_indices]
        mse_all = element_errors.mean(dim=1)
        best_c = mse_all.argmin(dim=-1)
        best_c_expanded = best_c.unsqueeze(1).expand(-1, 16)
        return value_table[element_indices, best_c_expanded], mse_all


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

    quantized_packed = quantizer._fakequant_layer_vanilla(layer_packed, weight_scale, weight_global_scale)
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

    bf16_weights = torch.randn(256, 16)
    opt_fp4, opt_scale = quantizer.fakequant_blocks_with_scale(bf16_weights)
    for i in range(opt_fp4.shape[0]):
        assert len(set(opt_fp4[i].tolist())) <= 4, f"Block {i} has more than 4 distinct values"
    assert (opt_scale > 0).all(), "Some scales are non-positive"
    opt_packed = quantizer.pack_fp4_to_uint8(opt_fp4.reshape(16, 256))
    assert opt_packed.dtype == torch.uint8, "Pack output must be uint8"

    scale_for_layer = torch.rand(out_features, in_features // 16).to(torch.float8_e4m3fn).to(torch.float32).clamp(min=0.01)
    gscale_for_layer = torch.tensor([2.0], dtype=torch.float32)
    opt_packed_layer, opt_scale_layer = quantizer.fakequant_layer(
        layer_packed, scale_for_layer, gscale_for_layer
    )
    opt_layer_fp4 = quantizer.unpack_uint8_to_fp4(opt_packed_layer)
    opt_layer_blocks = opt_layer_fp4.reshape(-1, 16)
    for i in range(opt_layer_blocks.shape[0]):
        assert len(set(opt_layer_blocks[i].tolist())) <= 4, f"Layer block {i} has more than 4 distinct values"
    assert tuple(opt_scale_layer.shape) == (out_features, in_features // 16), "Scale shape mismatch"

    print("All fakequant.py self-tests passed.")
