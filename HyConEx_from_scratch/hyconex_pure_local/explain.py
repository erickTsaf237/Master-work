from __future__ import annotations

import numpy as np
import torch

from hyconex_from_scratch.trainer import HyConExTrainer
from hyconex_pure_local.model import HyConExLocalModel


def explain_local_hypernet(
    trainer: HyConExTrainer,
    x: np.ndarray,
    sample_idx: int,
    *,
    class_names: list[str],
    class_idx: int | None = None,
    top_k: int = 12,
    y_true: int | None = None,
) -> dict:
    """
    Explication par classifieur local W(z)·z + b(z).

    Retourne les termes w_ci * z_i pour la classe expliquée et la classe rivale.
    """
    assert trainer.model is not None
    model = trainer.model
    if not hasattr(model, "local_hypernet_pack"):
        raise TypeError("Le modele doit exposer local_hypernet_pack()")

    x_row = np.asarray(x[sample_idx : sample_idx + 1], dtype=np.float32)
    x_t = torch.tensor(x_row, device=trainer.device)

    with torch.no_grad():
        pack = model.local_hypernet_pack(x_t)
        proba = torch.softmax(pack.logits, dim=1)
        y_pred = int(proba.argmax(dim=1).item())
        explain_c = int(class_idx if class_idx is not None else y_pred)

        z = pack.z[0].cpu().numpy()
        w_pred = pack.weights[0, explain_c].cpu().numpy()
        b_pred = float(pack.bias[0, explain_c].item())
        contrib_pred = pack.contributions[0, explain_c].cpu().numpy()
        logit_pred = float(pack.logits[0, explain_c].item())

        all_logits = pack.logits[0].cpu().numpy()
        rival_c = int(np.argsort(all_logits)[-2]) if model.num_classes > 1 else explain_c
        w_rival = pack.weights[0, rival_c].cpu().numpy()
        contrib_rival = pack.contributions[0, rival_c].cpu().numpy()
        logit_rival = float(pack.logits[0, rival_c].item())

    latent_dim = len(z)
    terms_pred = [
        {
            "dim": f"z_{i}",
            "z": float(z[i]),
            "w": float(w_pred[i]),
            "contribution": float(contrib_pred[i]),
        }
        for i in range(latent_dim)
    ]
    terms_pred.sort(key=lambda t: abs(t["contribution"]), reverse=True)

    terms_rival = [
        {
            "dim": f"z_{i}",
            "z": float(z[i]),
            "w": float(w_rival[i]),
            "contribution": float(contrib_rival[i]),
        }
        for i in range(latent_dim)
    ]
    terms_rival.sort(key=lambda t: abs(t["contribution"]), reverse=True)

    summary = (
        f"logit({class_names[explain_c]}) = "
        + " + ".join(f"{t['contribution']:+.3f}" for t in terms_pred[:5])
        + f" + ... + b={b_pred:+.3f} = {logit_pred:+.3f}"
    )

    return {
        "sample_idx": sample_idx,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_pred_name": class_names[y_pred],
        "explained_class": explain_c,
        "explained_class_name": class_names[explain_c],
        "logit_explained": logit_pred,
        "bias": b_pred,
        "rival_class": rival_c,
        "rival_class_name": class_names[rival_c],
        "logit_rival": logit_rival,
        "margin_logit": logit_pred - logit_rival if explain_c != rival_c else 0.0,
        "top_terms_explained": terms_pred[:top_k],
        "top_terms_rival": terms_rival[:top_k],
        "summary_text": summary,
    }


def explain_input_bridge(
    trainer: HyConExTrainer,
    x: np.ndarray,
    sample_idx: int,
    *,
    feature_names: list[str],
    class_names: list[str],
    class_idx: int | None = None,
    top_k: int = 12,
) -> dict:
    """
    Pont latent -> entrée : importance_j ≈ Σ_i |w_ci · ∂z_i/∂x_j| pour la classe expliquée.
    """
    assert trainer.model is not None
    model = trainer.model
    if not hasattr(model, "local_hypernet_pack"):
        raise TypeError("Le modele doit exposer local_hypernet_pack()")

    x_row = np.asarray(x[sample_idx : sample_idx + 1], dtype=np.float32)
    model.eval()

    with torch.no_grad():
        y_pred = int(model(torch.tensor(x_row, device=trainer.device)).argmax().item())
    explain_c = int(class_idx if class_idx is not None else y_pred)

    x_t = torch.tensor(x_row, device=trainer.device, requires_grad=True)
    logit_c = model.local_hypernet_pack(x_t).logits[0, explain_c]
    logit_c.backward()
    grad_x = x_t.grad.detach().cpu().numpy()[0]

    abs_g = np.abs(grad_x)
    top_idx = np.argsort(abs_g)[::-1][:top_k]
    attrs = [
        {
            "feature": feature_names[i],
            "importance": float(grad_x[i]),
            "value": float(x_row[0, i]),
        }
        for i in top_idx
    ]

    return {
        "sample_idx": sample_idx,
        "explained_class": explain_c,
        "explained_class_name": class_names[explain_c],
        "y_pred_name": class_names[y_pred],
        "top_features": attrs,
    }


def explain_counterfactual(
    trainer: HyConExTrainer,
    x: np.ndarray,
    sample_idx: int,
    target_class: int,
    *,
    feature_names: list[str],
    class_names: list[str],
    y_true: int | None = None,
    top_k: int = 12,
) -> dict:
    assert trainer.model is not None
    x_row = np.asarray(x[sample_idx : sample_idx + 1], dtype=np.float32)
    x_t = torch.tensor(x_row, device=trainer.device)
    y_tgt = torch.tensor([target_class], dtype=torch.long, device=trainer.device)

    with torch.no_grad():
        logits = trainer.model(x_t)
        proba = torch.softmax(logits, dim=1)
        y_pred = int(proba.argmax(dim=1).item())
        proba_orig = float(proba[0, y_pred].item())
        x_cf = trainer.model.generate_counterfactual(x_t, y_tgt)
        logits_cf = trainer.model(x_cf)
        proba_cf = torch.softmax(logits_cf, dim=1)
        y_cf = int(proba_cf.argmax(dim=1).item())
        proba_cf_v = float(proba_cf[0, y_cf].item())

    delta = (x_cf - x_t).detach().cpu().numpy()[0]
    x_np = x_row[0]
    changes = []
    for j in range(len(delta)):
        if abs(float(delta[j])) > 1e-4:
            changes.append(
                {
                    "feature": feature_names[j],
                    "from": float(x_np[j]),
                    "to": float(x_np[j] + delta[j]),
                    "delta": float(delta[j]),
                }
            )
    changes.sort(key=lambda c: abs(c["delta"]), reverse=True)

    return {
        "sample_idx": sample_idx,
        "y_true": y_true,
        "y_pred_orig": y_pred,
        "y_pred_orig_name": class_names[y_pred],
        "proba_orig": proba_orig,
        "y_target": target_class,
        "y_target_name": class_names[target_class],
        "y_pred_cf": y_cf,
        "valid": y_cf == target_class,
        "n_changes": len(changes),
        "changes": changes[:top_k],
    }
