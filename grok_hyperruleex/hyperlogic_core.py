"""Couche règles + OR HyperLogic et extraction IF-THEN (rapport §B)."""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn.functional as F


def smooth_rule_gate(inner: torch.Tensor, tau: float) -> torch.Tensor:
    """h(u) ≈ 1{u=0} via exp(-u²/τ)."""
    return torch.exp(-(inner * inner) / max(tau, 1e-6))


def hyperlogic_f(
    x: torch.Tensor, w: torch.Tensor, u: torch.Tensor, tau: float
) -> torch.Tensor:
    """
    x: (B, D) ±1
    w: (B, D, K)
    u: (B, K)
    retourne f(x): (B,)
    """
    inner = torch.einsum("bd,bdk->bk", x, w) - w.abs().sum(dim=1)
    ok = smooth_rule_gate(inner, tau)
    return (ok * u).sum(dim=1)


def logits_from_f(f: torch.Tensor, n_classes: int) -> torch.Tensor:
    """Binaire: [-f, f]. Multi-classe: généralisation simple (f par classe) non incluse ici."""
    if n_classes == 2:
        return torch.stack([-f, f], dim=1)
    raise NotImplementedError("n_classes>2: étendre avec une matrice de scores par classe.")


def extract_if_then_rules(
    w: torch.Tensor,
    u: torch.Tensor,
    feature_names: List[str],
    *,
    u_threshold: float = 0.05,
    w_pos: float = 0.5,
    w_neg: float = -0.5,
) -> List[str]:
    """|u_k| > seuil ; w_dk > 0.5 → vrai, w_dk < -0.5 → faux."""
    if w.dim() == 3:
        w, u = w[0], u[0]
    w = w.detach().cpu()
    u = u.detach().cpu()
    lines: List[str] = []
    d, k = w.shape
    for j in range(k):
        if abs(float(u[j])) < u_threshold:
            continue
        parts = []
        for i in range(d):
            val = float(w[i, j])
            name = feature_names[i] if i < len(feature_names) else f"f{i}"
            if val > w_pos:
                parts.append(f"{name}")
            elif val < w_neg:
                parts.append(f"NOT({name})")
        if not parts:
            continue
        conj = " AND ".join(parts)
        lines.append(f"RULE_{j+1}: IF {conj} THEN score (u={float(u[j]):.3f})")
    return lines


def local_feature_importance(w: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Somme sur les règles : Σ_k |w_dk| |u_k|, forme (B, D) ou (D,) si entrée 2D."""
    squeeze_batch = False
    if w.dim() == 2:
        w = w.unsqueeze(0)
        u = u.unsqueeze(0)
        squeeze_batch = True
    out = (w.abs() * u.abs().unsqueeze(1)).sum(dim=2)
    if squeeze_batch:
        return out.squeeze(0)
    return out
