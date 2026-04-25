"""Lissage softmax des blocs catégoriels (T=0.01, comme HyConEx one_hot_encoder_torch)."""

from __future__ import annotations

from typing import List, Tuple

import torch


def softmax_categorical_blocks(
    x_cf: torch.Tensor,
    cat_group_slices: List[Tuple[int, int]],
    temperature: float = 0.01,
) -> torch.Tensor:
    """
    x_cf: (N, D)
    cat_group_slices: liste (start, length) pour chaque variable catégorielle one-hot.
    """
    if not cat_group_slices:
        return x_cf
    out = x_cf.clone()
    t = max(temperature, 1e-8)
    for start, length in cat_group_slices:
        sl = out[:, start : start + length]
        out[:, start : start + length] = torch.softmax(sl / t, dim=-1)
    return out
