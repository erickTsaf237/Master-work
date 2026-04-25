"""
DR-Net-style rule layer + OR composition (NeurIPS HyperLogic paper),
with a compact HyperGAN-style mixer + two weight generators.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def dr_smooth_rules(
    x: torch.Tensor, w: torch.Tensor, u: torch.Tensor, tau: float
) -> torch.Tensor:
    """
    x: (B, D) in {-1, +1}
    w: (B, D, K)
    u: (B, K)
    returns f(x): (B,) pre-logit score for class Allow (positive = Allow).
    """
    wx = torch.einsum("bd,bdk->bk", x, w)
    abs_w = w.abs().sum(dim=1)
    inner = wx - abs_w
    ok = torch.exp(-(inner * inner) / tau)
    return (ok * u).sum(dim=1)


class MLPGenerator(nn.Module):
    def __init__(
        self, z_dim: int, hidden: List[int], out_dim: int, use_bn: bool = True
    ):
        super().__init__()
        layers: List[nn.Module] = []
        prev = z_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            if use_bn:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ELU())
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class HyperLogicMixer(nn.Module):
    """Maps Gaussian noise (B, s_dim) to per-generator latents (ngen, B, z_dim)."""

    def __init__(self, s_dim: int, z_dim: int, ngen: int, bias: bool = True):
        super().__init__()
        self.s_dim = s_dim
        self.z_dim = z_dim
        self.ngen = ngen
        self.fc1 = nn.Linear(s_dim, 512, bias=bias)
        self.fc2 = nn.Linear(512, 512, bias=bias)
        self.fc3 = nn.Linear(512, z_dim * ngen, bias=bias)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(512)

    def forward(self, eps: torch.Tensor) -> torch.Tensor:
        x = eps.view(-1, self.s_dim)
        x = x + torch.zeros_like(x).normal_(0, 0.01)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x).view(-1, self.ngen, self.z_dim)
        return x.permute(1, 0, 2)


class HyperLogicBranch(nn.Module):
    """
    Samples DR-Net weights via mixer+generators (M=1 Monte Carlo sample by default).
    Forward returns 2-class logits [logit_deny, logit_allow] for consistency with HyConEx.
    """

    def __init__(
        self,
        dim: int,
        n_rules: int,
        s_dim: int = 256,
        z_dim: int = 64,
        gen_hidden: Tuple[int, ...] = (512, 512),
        tau: float = 0.1,
        n_mc: int = 1,
    ):
        super().__init__()
        self.dim = dim
        self.n_rules = n_rules
        self.tau = tau
        self.n_mc = max(1, n_mc)
        self.mixer = HyperLogicMixer(s_dim, z_dim, ngen=2)
        gen_h = list(gen_hidden)
        self.gen_w = MLPGenerator(z_dim, hidden=gen_h, out_dim=dim * n_rules)
        self.gen_u = MLPGenerator(z_dim, hidden=gen_h, out_dim=n_rules)
        self.log_temperature = nn.Parameter(
            torch.log(torch.tensor(tau, dtype=torch.float32))
        )

    def _sample_weights(
        self, batch_size: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        eps = torch.randn(batch_size, self.mixer.s_dim, device=device)
        codes = self.mixer(eps)
        z_w = codes[0]
        z_u = codes[1]
        w_flat = self.gen_w(z_w)
        u_raw = self.gen_u(z_u)
        w = w_flat.view(batch_size, self.dim, self.n_rules)
        u = F.softplus(u_raw)
        return w, u

    def forward(
        self, x_pm: torch.Tensor, eps: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x_pm: (B, D) {-1, +1}
        returns logits (B, 2), w (B, D, K), u (B, K)
        """
        b, d = x_pm.shape
        device = x_pm.device
        tau = torch.exp(self.log_temperature).clamp_min(1e-4)
        f_sum = torch.zeros(b, device=device)
        w_last = torch.zeros(b, d, self.n_rules, device=device)
        u_last = torch.zeros(b, self.n_rules, device=device)
        for _ in range(self.n_mc):
            w, u = self._sample_weights(b, device)
            f = dr_smooth_rules(x_pm, w, u, float(tau.item()))
            f_sum = f_sum + f
            w_last, u_last = w, u
        f_mean = f_sum / self.n_mc
        logits = torch.stack([-f_mean, f_mean], dim=1)
        return logits, w_last, u_last


def extract_if_then_rules(
    w: torch.Tensor,
    u: torch.Tensor,
    feature_names: List[str],
    weight_threshold: float = 0.05,
) -> List[str]:
    """
    Build readable rules from DR-Net weights (single row / batch index 0).
    w: (D, K) or (1, D, K) — use .detach().cpu()
    u: (K,) rule strengths
    """
    if w.dim() == 3:
        w = w[0]
    if u.dim() == 2:
        u = u[0]
    w = w.detach().cpu()
    u = u.detach().cpu()
    lines: List[str] = []
    d, k = w.shape
    for j in range(k):
        active = []
        for i in range(d):
            val = float(w[i, j])
            if abs(val) < weight_threshold:
                continue
            name = feature_names[i] if i < len(feature_names) else f"f{i}"
            if val > 0:
                active.append(f"{name}>0")
            else:
                active.append(f"NOT({name}>0)")
        if not active:
            continue
        conj = " AND ".join(active)
        strength = float(u[j])
        lines.append(f"RULE_{j+1}: IF {conj} THEN Allow (w={strength:.3f})")
    return lines
