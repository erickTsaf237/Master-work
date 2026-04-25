"""HyperRuleEx : hyperréseau sur [x ; ε] → (w, u) et V (rapport §2–3)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from grok_hyperruleex.hyperlogic_core import hyperlogic_f, logits_from_f


@dataclass
class HyperRuleExConfig:
    dim: int
    n_rules: int
    n_classes: int = 2
    hidden: int = 256
    depth: int = 3
    tau: float = 0.1


class HyperRuleEx(nn.Module):
    """
    Entrée concaténée z = [x ; ε] ∈ ℝ^{2D}.
    Sorties : w ∈ ℝ^{D×K}, u ∈ ℝ^K, V ∈ ℝ^{C×D}.
    La prédiction provient uniquement de la branche HyperLogic sur f(x).
    """

    def __init__(self, cfg: HyperRuleExConfig):
        super().__init__()
        self.cfg = cfg
        d, k, c = cfg.dim, cfg.n_rules, cfg.n_classes
        in_dim = 2 * d
        layers: list[nn.Module] = []
        prev = in_dim
        for _ in range(cfg.depth):
            layers += [nn.Linear(prev, cfg.hidden), nn.LayerNorm(cfg.hidden), nn.GELU()]
            prev = cfg.hidden
        self.backbone = nn.Sequential(*layers)
        self.head_w = nn.Linear(prev, d * k)
        self.head_u = nn.Linear(prev, k)
        self.head_V = nn.Linear(prev, c * d)
        self.log_tau = nn.Parameter(torch.log(torch.tensor(cfg.tau, dtype=torch.float32)))

    def forward(
        self, x: torch.Tensor, eps: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        x: (B, D) en ±1
        eps: (B, D) bruit ; si None, tiré aléatoirement.
        """
        if eps is None:
            eps = torch.randn_like(x)
        z = torch.cat([x, eps], dim=-1)
        h = self.backbone(z)
        b, d, k = x.size(0), self.cfg.dim, self.cfg.n_rules
        c = self.cfg.n_classes
        w = self.head_w(h).view(b, d, k)
        u = F.softplus(self.head_u(h))
        V = self.head_V(h).view(b, c, d)
        tau = torch.exp(self.log_tau).clamp(1e-4, 10.0)
        f = hyperlogic_f(x, w, u, float(tau.item()))
        logits = logits_from_f(f, c)
        return {
            "logits": logits,
            "w": w,
            "u": u,
            "V": V,
            "f": f,
            "eps": eps,
        }
