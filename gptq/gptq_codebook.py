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
        use_importance: bool = True,
    ):
        self.in_features = in_features
        self.quantizer = quantizer or CodebookQuantizer()
        self.rel_damp = rel_damp
        self.block_size = block_size
        self.use_importance = use_importance
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
        h_diag = self.H.diag() if self.use_importance else None

        all_fp4 = torch.zeros_like(W)
        all_fp4_phase1 = torch.zeros_like(W)
        all_scales = torch.zeros(out_features, in_features // 16, device=W.device)

        codebook = self.quantizer.codebook.to(W.device)

        for c1 in range(0, in_features, self.block_size):
            c2 = min(c1 + self.block_size, in_features)
            w_blk = W[:, c1:c2].clone()
            errs = torch.zeros_like(w_blk)
            H_blk = H_inv_cho[c1:c2, c1:c2]

            for g in range(0, c2 - c1, 16):
                g_end = min(g + 16, c2 - c1)
                w_group = w_blk[:, g:g_end]

                group_importance = None
                if h_diag is not None:
                    group_importance = h_diag[c1 + g:c1 + g_end].unsqueeze(0)

                q_phase1, s_opt, best_k = self.quantizer.fakequant_blocks_with_scale(
                    w_group.reshape(-1, 16),
                    return_codebook_idx=True,
                    importance_weights=group_importance,
                )
                all_fp4_phase1[:, c1 + g:c1 + g_end] = q_phase1.reshape_as(w_group)
                s_opt = s_opt.reshape(out_features, 1)
                cb_vals = codebook[best_k]

                group_fp4 = torch.zeros(out_features, g_end - g, device=W.device)

                for j in range(g_end - g):
                    col = g + j
                    w_col = w_blk[:, col]
                    scaled_col = w_col / s_opt.squeeze(1)
                    dists = (scaled_col.unsqueeze(-1) - cb_vals).abs()
                    nearest_idx = dists.argmin(dim=-1)
                    q_col = cb_vals[torch.arange(out_features, device=W.device), nearest_idx]
                    group_fp4[:, j] = q_col

                    w_q_col = q_col * s_opt.squeeze(1)
                    d = H_blk[col, col]
                    err = (w_blk[:, col] - w_q_col) / d
                    if col + 1 < c2 - c1:
                        w_blk[:, col + 1:].addr_(err, H_blk[col, col + 1:], alpha=-1)
                    errs[:, col] = err

                all_fp4[:, c1 + g:c1 + g_end] = group_fp4
                all_scales[:, (c1 + g) // 16] = s_opt.reshape(out_features)

            if c2 < in_features:
                W[:, c2:].addmm_(errs, H_inv_cho[c1:c2, c2:], alpha=-1)

        W_q = all_fp4 * all_scales.repeat_interleave(16, dim=1)
        return all_fp4, all_scales, W_q, all_fp4_phase1

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
