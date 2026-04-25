"""Contre-factuels x'_m = x - α V_m avec recherche linéaire (rapport §C)."""

from __future__ import annotations

from typing import Callable, Tuple

import torch


def line_search_alpha(
    x: torch.Tensor,
    v_row: torch.Tensor,
    forward_logits: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    eps: torch.Tensor,
    target_class: int,
    *,
    alphas: torch.Tensor | None = None,
    dist_penalty: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Maximise softmax(logits)[m] - dist_penalty * ||x' - x||² sur une grille d'α.
    x: (D,), v_row: (D,), eps: (D,)
    retourne (alpha_best, x_cf)
    """
    if alphas is None:
        alphas = torch.linspace(0.0, 2.0, steps=25, device=x.device)
    best_score = torch.tensor(-1e9, device=x.device)
    best_alpha = alphas[0]
    best_xcf = x.clone()
    for a in alphas:
        x_cf = x - a * v_row
        logits = forward_logits(x_cf.unsqueeze(0), eps.unsqueeze(0)).squeeze(0)
        prob = torch.softmax(logits, dim=-1)[target_class]
        dist = (x_cf - x).pow(2).sum()
        score = prob - dist_penalty * dist
        if score > best_score:
            best_score = score
            best_alpha = a
            best_xcf = x_cf
    return best_alpha, best_xcf


def sample_eps_like_x(x: torch.Tensor) -> torch.Tensor:
    """ε ~ N(0, I) même dimension que x."""
    return torch.randn_like(x)
