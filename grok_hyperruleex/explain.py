"""interpret_rule, interpret_counterfactual, importances globales (rapport §C–D)."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch

from grok_hyperruleex.hyperlogic_core import local_feature_importance


def interpret_rule(rule_text: str, binary_to_original: Dict[int, str]) -> str:
    """Placeholders : remplace les indices f_i par la source si besoin."""
    _ = binary_to_original
    return rule_text


def interpret_counterfactual(
    x_cf: np.ndarray,
    x_orig: np.ndarray,
    feature_names: List[str],
) -> Dict[str, float]:
    """Différences par dimension (espace binaire ±1)."""
    d = {}
    for i, name in enumerate(feature_names):
        if i >= len(x_cf):
            break
        delta = float(x_cf[i] - x_orig[i])
        if abs(delta) > 1e-6:
            d[name] = delta
    return d


def global_feature_importance(
    model: torch.nn.Module,
    loader_x: torch.Tensor,
) -> np.ndarray:
    """Moyenne des contributions locales |w||u| sur un tenseur de batch."""
    model.eval()
    acc: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, loader_x.size(0), 64):
            xb = loader_x[i : i + 64]
            out = model(xb)
            imp = local_feature_importance(out["w"], out["u"])
            acc.append(imp.cpu().numpy())
    return np.concatenate(acc, axis=0).mean(axis=0)
