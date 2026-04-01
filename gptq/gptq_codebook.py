from __future__ import annotations

import math

import torch

from fakequant import CodebookQuantizer


class CodebookGPTQ:

    def __init__(
        self,
        in_features: int,
        quantizer: CodebookQuantizer | None = None,
        rel_damp: float = 1e-2,
        block_size: int = 128,
    ):
        self.in_features = in_features
        self.quantizer = quantizer or CodebookQuantizer()
        self.rel_damp = rel_damp
        self.block_size = block_size
        assert block_size % 16 == 0, "block_size must be a multiple of 16"

        self.H: torch.Tensor | None = None
        self.num_samples = 0

    @torch.no_grad()
    def update(self, X: torch.Tensor) -> None:
        if X.dim() == 3:
            X = X.reshape(-1, X.shape[-1])
        X = X.float()
        batch_size = X.shape[0]

        if self.H is None:
            self.H = torch.zeros(
                self.in_features, self.in_features,
                device=X.device, dtype=torch.float32,
            )

        beta = self.num_samples / (self.num_samples + batch_size)
        alpha = 2.0 / (self.num_samples + batch_size)
        self.H.mul_(beta)
        X_scaled = X.mul(math.sqrt(alpha))
        self.H.addmm_(X_scaled.T, X_scaled)
        self.num_samples += batch_size

    @torch.no_grad()
    def quantize(self, W: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self.H is not None and self.num_samples > 0, \
            "Must call update() with calibration data first"

        W = W.clone().float()
        out_features, in_features = W.shape
        assert in_features == self.in_features
        assert in_features % 16 == 0

        H_inv_cho = self._get_hessian_inverse()

        all_fp4 = torch.zeros_like(W)
        all_scales = torch.zeros(out_features, in_features // 16, device=W.device)

        for c1 in range(0, in_features, self.block_size):
            c2 = min(c1 + self.block_size, in_features)
            w_blk = W[:, c1:c2].clone()
            errs = torch.zeros_like(w_blk)
            H_blk = H_inv_cho[c1:c2, c1:c2]

            for g in range(0, c2 - c1, 16):
                g_end = min(g + 16, c2 - c1)
                w_group = w_blk[:, g:g_end]

                q_fp4, s_opt = self.quantizer.fakequant_blocks_with_scale(
                    w_group.reshape(-1, 16),
                )
                w_q = (q_fp4 * s_opt).reshape_as(w_group)

                all_fp4[:, c1 + g:c1 + g_end] = q_fp4.reshape_as(w_group)
                all_scales[:, (c1 + g) // 16] = s_opt.reshape(out_features)

                for j in range(g_end - g):
                    col = g + j
                    d = H_blk[col, col]
                    err = (w_blk[:, col] - w_q[:, j]) / d
                    if col + 1 < c2 - c1:
                        w_blk[:, col + 1:].addr_(err, H_blk[col, col + 1:], alpha=-1)
                    errs[:, col] = err

            if c2 < in_features:
                W[:, c2:].addmm_(errs, H_inv_cho[c1:c2, c2:], alpha=-1)

        W_q = all_fp4 * all_scales.repeat_interleave(16, dim=1)
        return all_fp4, all_scales, W_q

    @torch.no_grad()
    def _get_hessian_inverse(self) -> torch.Tensor:
        assert self.H is not None
        H = self.H.clone()

        dead = (H.diag() == 0).nonzero().squeeze(-1)
        if dead.numel() > 0:
            H[dead, :] = 0
            H[:, dead] = 0
            H[dead, dead] = 1

        damp = self.rel_damp * H.diag().mean()
        H.diagonal().add_(damp)

        try:
            H_inv = torch.cholesky_inverse(torch.linalg.cholesky(H))
            H_inv_cho = torch.linalg.cholesky(H_inv, upper=True)
        except Exception:
            H_inv_cho = torch.eye(self.in_features, device=H.device, dtype=torch.float32)

        return H_inv_cho
