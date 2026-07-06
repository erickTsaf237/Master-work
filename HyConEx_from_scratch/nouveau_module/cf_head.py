from __future__ import annotations

import torch
import torch.nn.functional as F


def unpack_cf_params(
    theta: torch.Tensor,
    input_dim: int,
    num_classes: int,
    hidden_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batched = theta.dim() == 2
    if not batched:
        theta = theta.unsqueeze(0)

    bsz = theta.shape[0]
    cf_in = input_dim + num_classes
    idx = 0

    s1 = cf_in * hidden_dim
    w1 = theta[:, idx : idx + s1].view(bsz, cf_in, hidden_dim)
    idx += s1

    b1 = theta[:, idx : idx + hidden_dim]
    idx += hidden_dim

    s2 = hidden_dim * input_dim
    w2 = theta[:, idx : idx + s2].view(bsz, hidden_dim, input_dim)
    idx += s2

    b2 = theta[:, idx : idx + input_dim]

    if not batched:
        return w1.squeeze(0), b1.squeeze(0), w2.squeeze(0), b2.squeeze(0)
    return w1, b1, w2, b2


def _straight_through_bipolar(p: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    soft = 2 * p - 1
    hard = torch.where(p > threshold, torch.ones_like(p), -torch.ones_like(p))
    return hard + soft - soft.detach()


def generate_cf_binary(
    x_bin: torch.Tensor,
    y_target: torch.Tensor,
    theta_cf: torch.Tensor,
    input_dim: int,
    num_classes: int,
    hidden_dim: int,
) -> torch.Tensor:
    w1, b1, w2, b2 = unpack_cf_params(theta_cf, input_dim, num_classes, hidden_dim)
    y_oh = F.one_hot(y_target, num_classes=num_classes).float()
    inp = torch.cat([x_bin, y_oh], dim=1)

    if w1.dim() == 3:
        h = torch.relu(torch.bmm(inp.unsqueeze(1), w1).squeeze(1) + b1)
        logits = torch.bmm(h.unsqueeze(1), w2).squeeze(1) + b2
    else:
        h = torch.relu(inp @ w1 + b1)
        logits = h @ w2 + b2

    p_flip = torch.sigmoid(logits)
    return _straight_through_bipolar(p_flip, threshold=0.5)
