"""NoiseHyConEx : W_final = sigmoid(a)*W_hyper(eps) + (1-sig)*W_main ; logits = x_aug @ W."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from .config import NoiseHyConExConfig
from .hypergenerator import HyperGenerator


class NoiseHyConEx(nn.Module):
    def __init__(self, cfg: NoiseHyConExConfig):
        super().__init__()
        self.cfg = cfg
        d, c = cfg.nr_features, cfg.nr_classes
        self.generator = HyperGenerator(
            noise_dim=cfg.noise_dim,
            nr_features=d,
            nr_classes=c,
            hidden_size=cfg.hidden_size,
            n_res_blocks=cfg.n_res_blocks,
            dropout_rate=cfg.dropout_rate,
        )
        self.W_main = nn.Parameter(torch.zeros(d + 1, c))
        nn.init.xavier_uniform_(self.W_main)
        self.raw_alpha = nn.Parameter(torch.zeros(()))

    def fuse_weights(self, W_hyper: torch.Tensor) -> torch.Tensor:
        """W_hyper (B,D+1,C) + W_main (D+1,C) -> W_final (B,D+1,C)."""
        alpha = torch.sigmoid(self.raw_alpha)
        return alpha * W_hyper + (1.0 - alpha) * self.W_main.unsqueeze(0)

    def _forward_core(
        self, x: torch.Tensor, eps: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """logits, W_hyper, W_final, x_aug."""
        b = x.size(0)
        x = x.view(b, self.cfg.nr_features)
        W_hyper = self.generator(eps)
        W_final = self.fuse_weights(W_hyper)
        x_aug = torch.cat([x, torch.ones(b, 1, device=x.device, dtype=x.dtype)], dim=1)
        logits = torch.einsum("bd,bdc->bc", x_aug, W_final)
        return logits, W_hyper, W_final, x_aug

    def forward(
        self,
        x: torch.Tensor,
        eps: Optional[torch.Tensor] = None,
        *,
        return_weights: bool = False,
        simple_weights: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, D)
        eps: (B, noise_dim) ou None -> tirage N(0,I)
        """
        b = x.size(0)
        if eps is None:
            eps = torch.randn(b, self.cfg.noise_dim, device=x.device, dtype=x.dtype)
        logits, _, W_final, x_aug = self._forward_core(x, eps)
        if not return_weights:
            return logits
        w_attr = W_final
        if not self.training and not simple_weights:
            repeated = torch.stack([x_aug for _ in range(self.cfg.nr_classes)], dim=2)
            w_attr = repeated[:, :-1, :] * W_final[:, :-1, :]
        return logits, w_attr

    def forward_dict(
        self,
        x: torch.Tensor,
        eps: Optional[torch.Tensor] = None,
        *,
        simple_weights: bool = True,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Retourne logits, poids d'attribution / W_final, W_hyper et eps (plan)."""
        b = x.size(0)
        if eps is None:
            eps = torch.randn(b, self.cfg.noise_dim, device=x.device, dtype=x.dtype)
        logits, W_hyper, W_final, x_aug = self._forward_core(x, eps)
        w_out = W_final
        if not self.training and not simple_weights:
            repeated = torch.stack([x_aug for _ in range(self.cfg.nr_classes)], dim=2)
            w_out = repeated[:, :-1, :] * W_final[:, :-1, :]
        return {
            "logits": logits,
            "weights": w_out,
            "W_hyper": W_hyper,
            "W_final": W_final,
            "eps": eps,
        }


if __name__ == "__main__":
    cfg = NoiseHyConExConfig(nr_features=12, nr_classes=3, noise_dim=32, n_res_blocks=2)
    m = NoiseHyConEx(cfg)
    x = torch.randn(4, 12)
    eps = torch.randn(4, cfg.noise_dim)
    logits, w = m(x, eps, return_weights=True, simple_weights=True)
    assert logits.shape == (4, 3) and w.shape == (4, 13, 3), (logits.shape, w.shape)
    d = m.forward_dict(x, eps)
    assert d["W_hyper"].shape == (4, 13, 3) and d["W_final"].shape == (4, 13, 3)
    print("Shapes OK:", logits.shape, w.shape)
