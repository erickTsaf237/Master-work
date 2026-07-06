from __future__ import annotations

import re
from typing import Any

import numpy as np
import torch

from hyconex_from_scratch.trainer import HyConExTrainer
from prepare_dlbac_datasets import format_rule

_Z_LITERAL_RE = re.compile(r"^z_(\d+)=(\+1|-1)$")


def parse_z_literal(literal: str) -> tuple[int, int] | None:
    """Parse 'z_48=+1' -> (48, +1)."""
    m = _Z_LITERAL_RE.match(literal.strip())
    if not m:
        return None
    return int(m.group(1)), (+1 if m.group(2) == "+1" else -1)


def compute_z_input_jacobian(
    model: torch.nn.Module,
    x_row: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """
    Jacobian J[i, j] = d z_i / d x_j au point x (1 x input_dim).
    """
    x_arr = np.asarray(x_row, dtype=np.float32).reshape(1, -1)
    x_t = torch.tensor(x_arr, device=device, requires_grad=True)
    z = model.encoder(x_t)
    n_latent = int(z.shape[1])
    n_input = int(x_t.shape[1])
    jac = np.zeros((n_latent, n_input), dtype=np.float32)

    for i in range(n_latent):
        if x_t.grad is not None:
            x_t.grad.zero_()
        z[0, i].backward(retain_graph=i < n_latent - 1)
        jac[i] = x_t.grad[0].detach().cpu().numpy()

    return jac


def decode_z_dimension(
    jac_row: np.ndarray,
    z_idx: int,
    literal_sign: int,
    *,
    feature_names: list[str],
    x_values: np.ndarray,
    top_k: int = 8,
    min_abs_grad: float = 1e-6,
) -> list[dict[str, Any]]:
    """
    Relie z_idx au sens du litteral (+1 / -1) aux colonnes oh_* les plus influentes.

    literal_sign=+1  -> oh_* avec gradient positif (poussent z_i vers le haut)
    literal_sign=-1  -> oh_* avec gradient negatif (poussent z_i vers le bas)
    """
    scores = jac_row if literal_sign > 0 else -jac_row
    order = np.argsort(scores)[::-1]

    out: list[dict[str, Any]] = []
    for j in order:
        if len(out) >= top_k:
            break
        g = float(jac_row[j])
        s = float(scores[j])
        if s < min_abs_grad:
            break
        out.append(
            {
                "z_dim": f"z_{z_idx}",
                "literal_sign": literal_sign,
                "feature": feature_names[j],
                "x_value": float(x_values[j]),
                "gradient_dz_dx": g,
                "alignment_score": s,
            }
        )
    return out


def decode_rule_literals(
    rule: dict,
    jac: np.ndarray,
    *,
    feature_names: list[str],
    x_values: np.ndarray,
    top_k_per_literal: int = 6,
) -> list[dict[str, Any]]:
    """Decode chaque litteral z_k=+/-1 de la regle."""
    decoded_literals: list[dict[str, Any]] = []
    for lit in rule.get("if", []):
        parsed = parse_z_literal(lit)
        if parsed is None:
            continue
        z_idx, sign = parsed
        if z_idx >= jac.shape[0]:
            continue
        tops = decode_z_dimension(
            jac[z_idx],
            z_idx,
            sign,
            feature_names=feature_names,
            x_values=x_values,
            top_k=top_k_per_literal,
        )
        decoded_literals.append(
            {
                "literal": lit,
                "z_dim": f"z_{z_idx}",
                "sign": sign,
                "top_oh_features": tops,
            }
        )
    return decoded_literals


def aggregate_rule_oh_features(
    decoded_literals: list[dict[str, Any]],
    *,
    top_k: int = 12,
) -> list[dict[str, Any]]:
    """Fusionne les oh_* de tous les litteraux (score max par feature)."""
    best: dict[str, dict[str, Any]] = {}
    for block in decoded_literals:
        for item in block.get("top_oh_features", []):
            feat = item["feature"]
            if feat not in best or item["alignment_score"] > best[feat]["alignment_score"]:
                best[feat] = {**item, "from_literal": block["literal"]}
    merged = sorted(best.values(), key=lambda d: d["alignment_score"], reverse=True)
    return merged[:top_k]


def decode_rule_for_sample(
    trainer: HyConExTrainer,
    rule: dict,
    x: np.ndarray,
    sample_idx: int,
    *,
    feature_names: list[str],
    top_k_per_literal: int = 6,
    top_k_merged: int = 12,
) -> dict[str, Any]:
    """Decode une regle DR-Net (sur z) vers oh_* pour un echantillon donne."""
    assert trainer.model is not None
    x_row = np.asarray(x[sample_idx : sample_idx + 1], dtype=np.float32)
    jac = compute_z_input_jacobian(trainer.model, x_row[0], trainer.device)
    decoded_literals = decode_rule_literals(
        rule,
        jac,
        feature_names=feature_names,
        x_values=x_row[0],
        top_k_per_literal=top_k_per_literal,
    )
    merged = aggregate_rule_oh_features(decoded_literals, top_k=top_k_merged)
    return {
        "sample_idx": sample_idx,
        "rule_text": format_rule(rule),
        "then_class": rule.get("then_class"),
        "score": rule.get("score"),
        "decoded_literals": decoded_literals,
        "top_oh_features_merged": merged,
    }


def format_decoded_rule(decoded: dict[str, Any]) -> str:
    """Texte lisible : regle z + traduction oh_*."""
    lines = [decoded.get("rule_text", "")]
    lines.append("  Decodage oh_* (echantillon {}):".format(decoded.get("sample_idx", "?")))
    for item in decoded.get("top_oh_features_merged", [])[:8]:
        lines.append(
            "    - {} (x={:.0f}, grad={:+.4f}, align={:.4f}) via {}".format(
                item["feature"],
                item["x_value"],
                item["gradient_dz_dx"],
                item["alignment_score"],
                item.get("from_literal", "?"),
            )
        )
    return "\n".join(lines)


def decode_rules_batch(
    trainer: HyConExTrainer,
    rules: list[dict],
    x: np.ndarray,
    sample_idx: int,
    *,
    feature_names: list[str],
    max_rules: int = 5,
) -> list[dict[str, Any]]:
    """Decode les N premieres regles pour un echantillon."""
    return [
        decode_rule_for_sample(
            trainer, rule, x, sample_idx, feature_names=feature_names,
        )
        for rule in rules[:max_rules]
    ]
