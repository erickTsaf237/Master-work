"""Build explanations from a single DualExplainModel forward pass."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

import torch

from .hyperlogic_drnet import extract_if_then_rules


class ExplanationPersona(str, Enum):
    END_USER = "end_user"
    AUDITOR = "auditor"
    ADMIN = "admin"


def _hyconex_row_weights(weights: torch.Tensor) -> torch.Tensor:
    """(B, D+1, K) -> (B, K, D+1)"""
    return weights.permute(0, 2, 1)


def counterfactual_to_target(
    x: torch.Tensor,
    weights: torch.Tensor,
    target_class: int,
    *,
    use_distance: bool = True,
    eps: float = 1.0,
) -> torch.Tensor:
    """
    One step toward target class m (HyConEx local linear geometry).
    x: (B, D)
    weights: (B, D+1, K) from HyperNet(simple_weights=True)
    """
    w_row = _hyconex_row_weights(weights)
    w_sel = w_row[:, target_class, :]
    w_feat, b = w_sel[:, :-1], w_sel[:, -1:]
    norm = torch.linalg.norm(w_feat, dim=-1, keepdim=True).clamp_min(1e-8)
    w_unit = w_feat / norm
    dist = (torch.sum(x * w_feat, dim=-1, keepdim=True) + b) / norm
    step = (dist * w_unit * eps) if use_distance else (w_feat * eps)
    return x - step


def feature_importance_scores(
    weights: torch.Tensor, predicted_class: int, batch_index: int = 0
) -> torch.Tensor:
    """Absolute linear coefficients as local importance (B, D)."""
    w_row = _hyconex_row_weights(weights)
    w_feat = w_row[batch_index, predicted_class, :-1]
    return w_feat.abs()


def build_explanation_bundle(
    forward_out: Dict[str, Any],
    x_hyconex: torch.Tensor,
    x_hyperlogic_pm: torch.Tensor,
    feature_names: List[str],
    *,
    use_distance_cf: bool = True,
) -> Dict[str, Any]:
    """
    Rich bundle for persona filtering (no second forward).
    """
    logits_hc = forward_out["logits_hyconex"]
    logits_hl = forward_out["logits_hyperlogic"]
    w_hc = forward_out["weights_hyconex"]
    w_hl = forward_out["weights_rule"]
    u_hl = forward_out["weights_or"]

    pred_hc = int(torch.argmax(logits_hc, dim=-1)[0].item())
    pred_hl = int(torch.argmax(logits_hl, dim=-1)[0].item())
    probs_hc = torch.softmax(logits_hc, dim=-1)[0].tolist()
    probs_hl = torch.softmax(logits_hl, dim=-1)[0].tolist()

    num_classes = logits_hc.shape[-1]
    cfs = {}
    for m in range(num_classes):
        if m != pred_hc:
            cfs[m] = counterfactual_to_target(
                x_hyconex,
                w_hc,
                m,
                use_distance=use_distance_cf,
            )[0].detach().cpu()

    imps = feature_importance_scores(w_hc, pred_hc, 0).detach().cpu()
    imp_map = {
        feature_names[i]: float(imps[i].item())
        for i in range(min(len(feature_names), imps.numel()))
    }

    rules = extract_if_then_rules(w_hl, u_hl, feature_names)

    return {
        "prediction_hyconex_class": pred_hc,
        "prediction_hyperlogic_class": pred_hl,
        "probabilities_hyconex": probs_hc,
        "probabilities_hyperlogic": probs_hl,
        "counterfactuals_by_target_class": cfs,
        "feature_importance_hyconex": imp_map,
        "rules_text": rules,
        "x_hyconex_sample": x_hyconex[0].detach().cpu(),
        "x_hyperlogic_pm_sample": x_hyperlogic_pm[0].detach().cpu(),
    }


def format_for_persona(
    bundle: Dict[str, Any], persona: ExplanationPersona
) -> Dict[str, Any]:
    if persona == ExplanationPersona.END_USER:
        probs = bundle["probabilities_hyconex"]
        nc = len(probs)
        pred = bundle["prediction_hyconex_class"]
        if nc == 2:
            target = 1 - pred
        else:
            target = next(m for m in range(nc) if m != pred)
        if target in bundle["counterfactuals_by_target_class"]:
            cf = bundle["counterfactuals_by_target_class"][target]
        else:
            cf = None
        return {
            "persona": persona.value,
            "summary": "Actionable changes to flip the access decision (HyConEx geometry).",
            "counterfactual_features": cf,
        }
    if persona == ExplanationPersona.AUDITOR:
        return {
            "persona": persona.value,
            "rules": bundle["rules_text"],
            "probabilities_hyconex": bundle["probabilities_hyconex"],
            "probabilities_hyperlogic": bundle["probabilities_hyperlogic"],
            "prediction_hyconex": bundle["prediction_hyconex_class"],
            "prediction_hyperlogic": bundle["prediction_hyperlogic_class"],
        }
    return {"persona": persona.value, "full": bundle}
