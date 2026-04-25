"""Pertes L_HyperLogic, L_conEx, L_stability (rapport page 4)."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def L_bce(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, y.long())


def L_sparsity_u(u: torch.Tensor) -> torch.Tensor:
    return u.abs().mean()


def L_diversity_f(f_samples: torch.Tensor) -> torch.Tensor:
    """Pénalise une faible variance de f sur plusieurs ε (encourage diversité)."""
    return -f_samples.var(dim=0).mean()


def L_stability(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    n_noises: int = 3,
) -> torch.Tensor:
    """KL ou MSE entre prédictions propres vs bruitées (même x, ε différents)."""
    logits_ref = model(x, torch.randn_like(x))["logits"].detach()
    acc = []
    for _ in range(n_noises):
        acc.append(model(x, torch.randn_like(x))["logits"])
    stack = torch.stack(acc, dim=0)
    mean = stack.mean(dim=0)
    p = F.softmax(logits_ref, dim=-1)
    q = F.softmax(mean, dim=-1)
    return F.mse_loss(p, q)


def L_conex_simple(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    target_wrong_class: torch.Tensor,
    *,
    alpha: float = 0.5,
) -> torch.Tensor:
    """
    Version simplifiée sans MAF : valider que x' = x - α V_m prédit m,
    + proximité L2.
    """
    out = model(x)
    V = out["V"]
    b = x.size(0)
    loss = x.new_zeros(())
    for i in range(b):
        m = int(target_wrong_class[i].item())
        x_cf = x[i] - alpha * V[i, m]
        logits_cf = model(x_cf.unsqueeze(0), out["eps"][i].unsqueeze(0))["logits"].squeeze(0)
        loss = loss + F.cross_entropy(logits_cf.unsqueeze(0), torch.tensor([m], device=x.device))
        loss = loss + (x_cf - x[i]).pow(2).mean()
    return loss / b


def total_loss(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    *,
    lambda_sparse: float = 0.05,
    lambda_div: float = 0.02,
    lambda_stab: float = 0.05,
    lambda_cf: float = 0.0,
) -> Dict[str, torch.Tensor]:
    x, y = batch["x"], batch["y"]
    out = model(x)
    l_bce = L_bce(out["logits"], y)
    l_sp = L_sparsity_u(out["u"])
    # diversité : plusieurs ε → (w,u,f) différents via le hyperréseau
    m = min(4, x.size(0))
    x_sub = x[:m]
    fs = []
    for _ in range(3):
        oi = model(x_sub)
        fs.append(oi["f"])
    l_div = L_diversity_f(torch.stack(fs, dim=0))
    l_stab = L_stability(model, x, y)
    total = (
        l_bce
        + lambda_sparse * l_sp
        + lambda_div * l_div
        + lambda_stab * l_stab
    )
    if lambda_cf > 0.0 and "cf_target" in batch:
        total = total + lambda_cf * L_conex_simple(
            model, x, y, batch["cf_target"], alpha=0.3
        )
    return {
        "total": total,
        "bce": l_bce.detach(),
        "sparse": l_sp.detach(),
        "div": l_div.detach(),
        "stab": l_stab.detach(),
    }
