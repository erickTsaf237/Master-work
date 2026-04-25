"""Training objectives: HyConEx CE, HyperLogic CE, consistency, optional flow on CF."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

from .explain import counterfactual_to_target


def consistency_kl(
    logits_a: torch.Tensor, logits_b: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """Symmetric KL between softmax distributions (batch mean)."""
    p = F.softmax(logits_a, dim=-1)
    q = F.softmax(logits_b, dim=-1)
    kl_pq = (p * (p.clamp_min(eps).log() - q.clamp_min(eps).log())).sum(dim=-1).mean()
    kl_qp = (q * (q.clamp_min(eps).log() - p.clamp_min(eps).log())).sum(dim=-1).mean()
    return 0.5 * (kl_pq + kl_qp)


def hyconex_counterfactual_ce(
    model: Any,
    x_hyconex: torch.Tensor,
    forward_out: Dict[str, torch.Tensor],
    y: torch.Tensor,
    *,
    use_distance: bool = True,
) -> torch.Tensor:
    """
    Push counterfactuals toward alternative class (per HyConEx training idea), without MAF.
    For each sample, pick a random target class != y and minimize CE on that CF point.
    """
    logits_hc = forward_out["logits_hyconex"]
    w = forward_out["weights_hyconex"]
    device = x_hyconex.device
    B = x_hyconex.size(0)
    K = logits_hc.size(-1)
    loss = torch.zeros((), device=device)
    count = 0
    was_training = model.hyconex_net.training
    model.hyconex_net.eval()
    for b in range(B):
        yt = int(y[b].item())
        choices = [m for m in range(K) if m != yt]
        if not choices:
            continue
        m = int(torch.randint(len(choices), (1,), device=device).item())
        m = choices[m]
        xb = x_hyconex[b : b + 1]
        wb = w[b : b + 1]
        xcf = counterfactual_to_target(xb, wb, m, use_distance=use_distance)
        logits_cf, _ = model.hyconex_net(
            xcf, return_weights=True, simple_weights=True
        )
        t = torch.tensor([m], device=device, dtype=torch.long)
        loss = loss + F.cross_entropy(logits_cf, t)
        count += 1
    if was_training:
        model.hyconex_net.train()
    if count == 0:
        return loss
    return loss / count


def flow_penalty(
    flow: Optional[torch.nn.Module],
    x_cf: torch.Tensor,
    target_class: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    if flow is None:
        return torch.zeros((), device=x_cf.device)
    # HyConEx MaskedAutoregressiveFlow forwards to log_prob
    lp = flow(x_cf, context=target_class.view(-1, 1).float())
    return F.relu(threshold - lp.exp()).mean()


def combined_loss(
    model: Any,
    x_hyconex: torch.Tensor,
    x_pm: torch.Tensor,
    forward_out: Dict[str, torch.Tensor],
    *,
    y: torch.Tensor,
    lambda_consistency: float = 0.1,
    lambda_cf: float = 0.0,
    lambda_flow: float = 0.0,
    flow: Optional[torch.nn.Module] = None,
    flow_threshold: float = 0.0,
    use_distance_cf: bool = True,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    forward_out from model.forward(x_hyconex, x_pm).
    x_pm is unused in the default loss but kept for API symmetry with optional extensions.
    """
    _ = x_pm  # reserved for future joint density terms
    logits_hc = forward_out["logits_hyconex"]
    logits_hl = forward_out["logits_hyperlogic"]

    l_hc = F.cross_entropy(logits_hc, y)
    l_hl = F.cross_entropy(logits_hl, y)
    l_cons = consistency_kl(logits_hc, logits_hl)

    l_cf = torch.zeros((), device=y.device)
    if lambda_cf > 0.0:
        l_cf = hyconex_counterfactual_ce(
            model,
            x_hyconex,
            forward_out,
            y,
            use_distance=use_distance_cf,
        )

    l_flow = torch.zeros((), device=y.device)
    if lambda_flow > 0.0 and flow is not None:
        B = y.size(0)
        K = logits_hc.size(-1)
        w = forward_out["weights_hyconex"]
        for b in range(B):
            yt = int(y[b].item())
            for m in range(K):
                if m == yt:
                    continue
                xb = x_hyconex[b : b + 1]
                wb = w[b : b + 1]
                xcf = counterfactual_to_target(xb, wb, m, use_distance=use_distance_cf)
                tc = torch.tensor([[m]], device=y.device, dtype=torch.float32)
                l_flow = l_flow + flow_penalty(flow, xcf, tc, flow_threshold)
        denom = max(B * max(K - 1, 1), 1)
        l_flow = l_flow / denom

    total = (
        l_hc
        + l_hl
        + lambda_consistency * l_cons
        + lambda_cf * l_cf
        + lambda_flow * l_flow
    )
    metrics = {
        "loss_total": float(total.detach().item()),
        "loss_hyconex": float(l_hc.detach().item()),
        "loss_hyperlogic": float(l_hl.detach().item()),
        "loss_consistency": float(l_cons.detach().item()),
        "loss_cf": float(l_cf.detach().item()),
        "loss_flow": float(l_flow.detach().item()),
    }
    return total, metrics
