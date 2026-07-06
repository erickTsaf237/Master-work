"""Utilitaires RuleConEx : règles, contrefactuels, importances, explain_sample."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from nouveau_module.main_rule_net import extract_rules
from ruleconex.model import RuleConExForwardPack, RuleConExModel, split_enc_bipolar, to_bipolar


def numpy_to_model_input(x: np.ndarray) -> np.ndarray:
    """Assure entrée [0,1] pour DLBAC one-hot."""
    x = np.asarray(x, dtype=np.float32)
    if x.min() < -0.01:
        return ((x + 1.0) * 0.5).astype(np.float32)
    return x


@dataclass
class ExplanationReport:
    prediction: int
    prediction_label: str
    probabilities: dict[str, float]
    top_importances: list[tuple[str, float]]
    counterfactuals: list[dict[str, Any]]
    rules: list[dict[str, Any]]
    text_report: str


def feature_importances_from_pack(
    pack: RuleConExForwardPack,
    feature_names: list[str],
    class_idx: int | None = None,
    top_k: int = 15,
) -> list[tuple[str, float]]:
    imp = pack.input_importance.detach().cpu()
    if class_idx is None:
        scores = imp.mean(dim=(0, 1))
    else:
        scores = imp[:, class_idx, :].mean(dim=0)
    idx = torch.argsort(scores, descending=True)[:top_k]
    return [(feature_names[int(i)], float(scores[i])) for i in idx]


def extract_rules_from_pack(
    pack: RuleConExForwardPack,
    feature_names: list[str],
    class_names: list[str],
    *,
    rules_on_input: bool = True,
    latent_dim: int | None = None,
    top_per_rule: int = 4,
    min_abs_weight: float = 0.03,
    max_rules: int = 12,
) -> list[dict[str, Any]]:
    rule_names = feature_names if rules_on_input else [f"z_{i}" for i in range(latent_dim or 64)]
    w_rule, _, w_out, _ = pack.rule_params
    rules = extract_rules(
        w_rule,
        w_out,
        rule_names,
        class_names,
        top_per_rule=top_per_rule,
        min_abs_weight=min_abs_weight,
    )
    return rules[:max_rules]


def onehot_feature_flips(
    enc: np.ndarray,
    cf: np.ndarray,
    feature_names: list[str],
    *,
    max_flips: int = 15,
) -> list[dict[str, Any]]:
    """Vrais changements one-hot : colonne active/inactive (seuil 0.5)."""
    enc = np.asarray(enc, dtype=np.float32).ravel()
    cf = np.asarray(cf, dtype=np.float32).ravel()
    before = enc > 0.5
    after = cf > 0.5
    changed_idx = np.where(before != after)[0]
    if changed_idx.size == 0:
        changed_idx = np.where(np.abs(enc - cf) > 0.5)[0]

    flips: list[dict[str, Any]] = []
    for i in changed_idx[:max_flips]:
        fv, tv = float(enc[i]), float(cf[i])
        flips.append(
            {
                "feature": feature_names[int(i)],
                "from": "actif" if fv > 0.5 else "inactif",
                "to": "actif" if tv > 0.5 else "inactif",
                "from_val": fv,
                "to_val": tv,
            }
        )
    return flips


def format_rules_text(rules: list[dict[str, Any]], *, max_rules: int = 20) -> str:
    """Affiche les règles IF-THEN en texte lisible."""
    if not rules:
        return "(aucune règle extraite — réduire min_abs_weight ou augmenter num_rules)"
    lines = ["=" * 60, "Règles IF-THEN extraites (HyperLogic)", "=" * 60]
    for r in rules[:max_rules]:
        cond = " AND ".join(r["if"]) if r["if"] else "(vide)"
        lines.append(f"R{r['rule_id']:02d} | IF {cond}")
        lines.append(f"     THEN {r['then_class']}  (confiance={r['score']:.3f})")
        lines.append("")
    return "\n".join(lines)


def counterfactual_report(
    model: RuleConExModel,
    x: np.ndarray | torch.Tensor,
    y_true: int,
    feature_names: list[str],
    class_names: list[str],
    *,
    device: torch.device | None = None,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    device = device or next(model.parameters()).device
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(numpy_to_model_input(np.asarray(x)), dtype=torch.float32, device=device)
    else:
        x = x.to(device).float()
    if x.dim() == 1:
        x = x.unsqueeze(0)

    pack = model.forward_pack(x)
    pred = int(pack.logits.argmax(dim=1).item())

    reports: list[dict[str, Any]] = []

    for c in range(model.num_classes):
        if c == pred:
            continue
        y_t = torch.tensor([c], device=device, dtype=torch.long)
        x_cf = model.generate_counterfactual(x, y_t, pack=pack)
        logits_cf = model(x_cf)
        prob_cf = torch.softmax(logits_cf, dim=1)[0, c].item()

        enc_in = pack.enc_in[0].detach().cpu().numpy()
        cf_np = x_cf[0].detach().cpu().numpy()
        flips = onehot_feature_flips(enc_in, cf_np, feature_names)

        reports.append(
            {
                "prediction": class_names[pred],
                "prediction_idx": pred,
                "true_label": class_names[int(y_true)] if 0 <= y_true < len(class_names) else str(y_true),
                "target_class": class_names[c],
                "target_idx": c,
                "cf_success_prob": float(prob_cf),
                "n_flips": len(flips),
                "flipped_features": flips,
            }
        )

    reports.sort(key=lambda r: r["cf_success_prob"], reverse=True)
    return reports[:top_k]


def explain_sample(
    model: RuleConExModel,
    user_meta: np.ndarray,
    resource_meta: np.ndarray | None,
    *,
    feature_names: list[str],
    class_names: list[str],
    device: torch.device | None = None,
    top_k: int = 12,
) -> ExplanationReport:
    """
    Rapport texte + structures pour figures.

    user_meta / resource_meta : vecteurs one-hot ou concaténés [0,1].
    Si resource_meta est None, user_meta est le vecteur complet.
    """
    device = device or next(model.parameters()).device

    if resource_meta is not None:
        x_np = np.concatenate([np.asarray(user_meta).ravel(), np.asarray(resource_meta).ravel()])
    else:
        x_np = np.asarray(user_meta).ravel()

    x = torch.tensor(x_np, dtype=torch.float32, device=device).unsqueeze(0)
    pack = model.forward_pack(x)
    probs = torch.softmax(pack.logits, dim=1)[0]
    pred = int(probs.argmax().item())

    prob_dict = {class_names[i]: float(probs[i]) for i in range(len(class_names))}
    imps = feature_importances_from_pack(pack, feature_names, class_idx=pred, top_k=top_k)
    rules = extract_rules_from_pack(pack, feature_names, class_names)
    cfs = counterfactual_report(model, x, pred, feature_names, class_names, device=device)

    lines = [
        "=" * 60,
        "RuleConEx — Rapport d'explication",
        "=" * 60,
        f"Prédiction : {class_names[pred]} (idx={pred})",
        "Probabilités :",
    ]
    for name, p in prob_dict.items():
        lines.append(f"  - {name}: {p:.4f}")

    lines.append("\nTop importances locales (métadonnées) :")
    for fname, score in imps:
        lines.append(f"  - {fname}: {score:.4f}")

    lines.append("\nRègles IF-THEN extraites :")
    for r in rules[:8]:
        cond = " AND ".join(r["if"])
        lines.append(f"  IF {cond} THEN {r['then_class']} (score={r['score']:.3f})")

    lines.append("\nContrefactuels (changer la décision) :")
    if not cfs:
        lines.append("  (aucun — prédiction unique ou CF non générés)")
    for cf in cfs[:3]:
        lines.append(f"  → Cible {cf['target_class']} (P={cf['cf_success_prob']:.3f})")
        if cf["flipped_features"]:
            for flip in cf["flipped_features"][:5]:
                lines.append(
                    f"      {flip['feature']}: {flip['from']} → {flip['to']} "
                    f"({flip['from_val']:.3f} → {flip['to_val']:.3f})"
                )
        else:
            lines.append("      (aucune métadonnée modifiée au-dessus du seuil)")

    text = "\n".join(lines)
    return ExplanationReport(
        prediction=pred,
        prediction_label=class_names[pred],
        probabilities=prob_dict,
        top_importances=imps,
        counterfactuals=cfs,
        rules=rules,
        text_report=text,
    )
