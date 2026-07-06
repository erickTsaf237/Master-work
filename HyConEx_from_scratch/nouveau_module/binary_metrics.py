"""Métriques binaires (grant/deny) et recherche de seuil sur probabilités."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def predict_with_grant_threshold(proba: np.ndarray, threshold: float) -> np.ndarray:
    """Classe 1 = grant si proba[:, 1] >= threshold, sinon deny (0)."""
    if proba.shape[1] != 2:
        raise ValueError("Seuil grant/deny : attendu 2 colonnes de probabilités.")
    return (proba[:, 1] >= float(threshold)).astype(np.int64)


def deny_class_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Métriques pour la classe minoritaire deny (label 0)."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    return {
        "deny_precision": float(precision_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "deny_recall": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "deny_f1": float(f1_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "grant_recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


def tune_grant_threshold(
    proba: np.ndarray,
    y_true: np.ndarray,
    *,
    metric: str = "deny_f1",
    n_steps: int = 91,
) -> tuple[float, dict[str, Any]]:
    """
    Balaye le seuil sur P(grant) = proba[:, 1].

    metric: deny_f1 | deny_recall | balanced_accuracy
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    thresholds = np.linspace(0.05, 0.95, n_steps)
    best_t = 0.5
    best_score = -1.0
    best_pred: np.ndarray | None = None

    for t in thresholds:
        y_pred = predict_with_grant_threshold(proba, t)
        if metric == "deny_recall":
            score = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
        elif metric == "balanced_accuracy":
            score = balanced_accuracy_score(y_true, y_pred)
        else:
            score = f1_score(y_true, y_pred, pos_label=0, zero_division=0)

        if score > best_score:
            best_score = float(score)
            best_t = float(t)
            best_pred = y_pred

    assert best_pred is not None
    out: dict[str, Any] = {
        "threshold_grant_prob": best_t,
        "tune_metric": metric,
        "tune_score": best_score,
        "accuracy": float(accuracy_score(y_true, best_pred)),
        "confusion_matrix": confusion_matrix(y_true, best_pred).tolist(),
    }
    out.update(deny_class_metrics(y_true, best_pred))
    return best_t, out


def summarize_binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    proba: np.ndarray | None = None,
    *,
    threshold: float | None = None,
) -> dict[str, Any]:
    """Résumé standard + option seuil."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    summary: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    summary.update(deny_class_metrics(y_true, y_pred))
    if threshold is not None:
        summary["threshold_grant_prob"] = float(threshold)
    if proba is not None and proba.shape[1] == 2:
        summary["mean_proba_grant"] = float(proba[:, 1].mean())
        summary["mean_proba_grant_deny_only"] = float(proba[y_true == 0, 1].mean()) if np.any(y_true == 0) else None
        summary["mean_proba_grant_grant_only"] = float(proba[y_true == 1, 1].mean()) if np.any(y_true == 1) else None
    return summary
