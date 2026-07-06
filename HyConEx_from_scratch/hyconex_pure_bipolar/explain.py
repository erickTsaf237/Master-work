from __future__ import annotations

import numpy as np
import torch

from hyconex_from_scratch.trainer import HyConExTrainer
from hyconex_pure_local.explain import explain_input_bridge, explain_local_hypernet
from prepare_dlbac_datasets import format_rule


def explain_counterfactual_bipolar(
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
    from hyconex_pure_bipolar.bipolar import bipolar_to_continuous, continuous_to_bipolar

    x_row = np.asarray(continuous_to_bipolar(x[sample_idx : sample_idx + 1]), dtype=np.float32)
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

    x_np = x_row[0]
    x_cf_bin = np.asarray(continuous_to_bipolar(x_cf.detach().cpu().numpy()[0]), dtype=np.float32)
    flips = []
    for j in range(len(x_np)):
        if x_np[j] != x_cf_bin[j]:
            flips.append(
                {
                    "feature": feature_names[j],
                    "from": int(x_np[j]),
                    "to": int(x_cf_bin[j]),
                }
            )
    flips.sort(key=lambda c: abs(c["to"] - c["from"]), reverse=True)

    return {
        "sample_idx": sample_idx,
        "y_true": y_true,
        "y_pred_orig": y_pred,
        "y_pred_orig_name": class_names[y_pred] if y_pred < len(class_names) else str(y_pred),
        "proba_orig": proba_orig,
        "y_target": target_class,
        "y_target_name": class_names[target_class] if target_class < len(class_names) else str(target_class),
        "y_pred_cf": y_cf,
        "valid": y_cf == target_class,
        "n_flips": len(flips),
        "flips": flips[:top_k],
        "input_space": "bipolar",
    }


__all__ = [
    "explain_local_hypernet",
    "explain_input_bridge",
    "explain_counterfactual_bipolar",
    "format_rule",
]
