"""Utilitaires d'evaluation TabResNet DLBAC (notebook / scripts)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from hyconex_pure_bipolar import HyConExBipolarRulesTrainer
from nouveau_module import HybridDRTrainer
from tabresnet_dlbac import TabResNetDLBACConfig, TabResNetDLBACTrainer


def compute_classification_metrics(
    y_true: np.ndarray,
    proba: np.ndarray,
    *,
    class_names: list[str] | None = None,
) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.argmax(proba, axis=1)
    nc = proba.shape[1]

    out: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "log_loss": float(log_loss(y_true, np.clip(proba, 1e-7, 1 - 1e-7))),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    if nc == 2:
        out["auroc"] = float(roc_auc_score(y_true, proba[:, 1]))
        out["deny_f1"] = float(f1_score(y_true, y_pred, pos_label=0, zero_division=0))
        out["grant_f1"] = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
    else:
        out["auroc"] = float(roc_auc_score(y_true, proba, multi_class="ovr"))

    labels = class_names or [str(i) for i in range(nc)]
    out["classification_report"] = classification_report(
        y_true, y_pred, target_names=labels, output_dict=True, zero_division=0
    )
    return out


def load_trainer_from_checkpoint(
    spec_name: str,
    splits,
    results_dir: Path,
    *,
    device: str | torch.device = "auto",
) -> TabResNetDLBACTrainer:
    results_dir = Path(results_dir)
    cfg_path = results_dir / f"{spec_name}_config.json"
    ckpt_path = results_dir / f"{spec_name}_model.pt"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint introuvable: {ckpt_path}")

    if device in (None, "auto"):
        map_loc = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device_arg: str | torch.device = "auto"
    elif isinstance(device, torch.device):
        map_loc = device
        device_arg = device
    else:
        map_loc = torch.device(device)
        device_arg = device

    cfg = TabResNetDLBACConfig(**json.loads(cfg_path.read_text(encoding="utf-8")))
    ckpt = torch.load(ckpt_path, map_location=map_loc, weights_only=False)
    mode = ckpt["mode"]
    trainer = TabResNetDLBACTrainer(cfg, device=str(device_arg) if isinstance(device_arg, torch.device) else device_arg)
    trainer.class_names = ckpt.get("class_names", splits.class_names)
    trainer.feature_names = ckpt.get("feature_names", splits.feature_names)

    if mode == "instance":
        trainer.mode = "instance"
        trainer._hybrid = HybridDRTrainer(trainer._hybrid_config(), device=trainer._device_arg)
        trainer._hybrid.binarizer.fit_transform(splits.x_train, feature_names=splits.feature_names)
        dim = trainer._hybrid.binarizer.transform(splits.x_train[:1]).shape[1]
        trainer._hybrid._build_model(dim, splits.num_classes)
        trainer._hybrid.model.load_state_dict(ckpt["state_dict"])
        trainer._hybrid.model.eval()
        trainer._hybrid.class_names = trainer.class_names
        return trainer

    if mode == "bipolar_hyper":
        trainer.mode = "bipolar_hyper"
        bcfg = trainer._bipolar_config(splits.x_train.shape[1], splits.num_classes)
        trainer._bipolar = HyConExBipolarRulesTrainer(bcfg, device=trainer._device_arg)
        trainer._bipolar.feature_names = trainer.feature_names
        trainer._bipolar.class_names = trainer.class_names
        trainer._bipolar._ensure_model(splits.x_train.shape[1], splits.num_classes)
        trainer._bipolar.model.load_state_dict(ckpt["state_dict"])
        trainer._bipolar._num_classes = splits.num_classes
        trainer._bipolar.model.eval()
        return trainer

    raise ValueError(f"Mode checkpoint inconnu: {mode}")
