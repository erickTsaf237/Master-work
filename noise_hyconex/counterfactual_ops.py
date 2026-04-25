"""Construction des contre-factuels x_cf et masques (adapté de HyConEx model.py)."""

from __future__ import annotations

from typing import Tuple

import torch

from .categorical_smooth import softmax_categorical_blocks
from .dataset_adapter import TabularFeatureLayout


def get_counterfact_with_mask(
    x: torch.Tensor,
    output: torch.Tensor,
    weights: torch.Tensor,
    layout: TabularFeatureLayout,
    device: torch.device,
    *,
    use_projection: bool,
    cat_temperature: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Tuple[int, int, int]]:
    """
    weights: (B, D+1, C) comme retourné par NoiseHyConEx (après attribut si besoin;
             ici on attend les poids bruts W_final pour CF géométrique).
    """
    y_pred = torch.argmax(output, dim=-1)
    weights = weights.permute(0, 2, 1)
    w, bias = weights[:, :, :-1], weights[:, :, -1:]
    batch_sz, d_out, d_in = weights.shape
    x_exp = x.unsqueeze(1)

    distance = (torch.sum(x_exp * w, dim=-1, keepdim=True) + bias) / torch.linalg.norm(
        w, dim=-1, keepdim=True
    )
    w_unit = w / torch.linalg.norm(w, dim=-1, keepdim=True)
    x_cf = x_exp - distance * w_unit if use_projection else x_exp - w
    x_cf = x_cf.reshape(batch_sz * d_out, d_in - 1)
    x_cf = softmax_categorical_blocks(x_cf, layout.cat_group_slices, temperature=cat_temperature)

    mask = torch.arange(d_out, device=device).unsqueeze(0) != y_pred.unsqueeze(1)
    target_values = torch.arange(d_out, device=device).unsqueeze(0).expand(batch_sz, d_out)
    target_values = target_values.reshape(batch_sz * d_out)

    return x_cf, x_exp, mask, target_values, (batch_sz, d_out, d_in)
