from __future__ import annotations

import torch
import torch.nn.functional as F


def unpack_cf_params(theta: torch.Tensor, input_dim: int, num_classes: int, hidden_dim: int) -> tuple[torch.Tensor, ...]:
    cf_in = input_dim + num_classes
    idx = 0

    s1 = cf_in * hidden_dim
    w1 = theta[idx : idx + s1].view(cf_in, hidden_dim)
    idx += s1

    b1 = theta[idx : idx + hidden_dim]
    idx += hidden_dim

    s2 = hidden_dim * input_dim
    w2 = theta[idx : idx + s2].view(hidden_dim, input_dim)
    idx += s2

    b2 = theta[idx : idx + input_dim]
    return w1, b1, w2, b2


def _straight_through_binary(p: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    hard = (p > threshold).float()
    return hard + p - p.detach()


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

    h = torch.relu(inp @ w1 + b1)
    logits = h @ w2 + b2
    p_flip = torch.sigmoid(logits)

    x_cf = _straight_through_binary(p_flip, threshold=0.5)
    return x_cf
