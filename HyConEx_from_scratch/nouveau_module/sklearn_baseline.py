"""Baseline sklearn sur features continues (reference DLBAC Amazon)."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.svm import LinearSVC

from nouveau_module.binary_metrics import predict_with_grant_threshold, tune_grant_threshold


def train_histgb_baseline(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    seed: int = 42,
) -> dict[str, Any]:
    """Gradient boosting sur metadonnees MinMax (sans binarisation)."""
    clf = HistGradientBoostingClassifier(
        max_depth=8,
        learning_rate=0.08,
        max_iter=300,
        class_weight="balanced",
        random_state=seed,
    )
    x_train = np.asarray(x_train, dtype=np.float32)
    clf.fit(x_train, y_train)

    proba_val = clf.predict_proba(x_val)
    proba_test = clf.predict_proba(x_test)
    thresh, val_tune = tune_grant_threshold(proba_val, y_val, metric="deny_f1")

    pred_test = predict_with_grant_threshold(proba_test, thresh)
    pred_argmax = clf.predict(x_test)

    return {
        "model": "HistGradientBoosting",
        "grant_threshold": thresh,
        "val_tune": val_tune,
        "test_auroc": float(roc_auc_score(y_test, proba_test[:, 1])),
        "test_accuracy": float(accuracy_score(y_test, pred_argmax)),
        "test_deny_recall_argmax": float(recall_score(y_test, pred_argmax, pos_label=0, zero_division=0)),
        "test_deny_recall_tuned": float(recall_score(y_test, pred_test, pos_label=0, zero_division=0)),
        "test_deny_f1_tuned": float(f1_score(y_test, pred_test, pos_label=0, zero_division=0)),
        "test_balanced_accuracy_tuned": float(balanced_accuracy_score(y_test, pred_test)),
    }


def train_linear_svm_baseline(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    seed: int = 42,
) -> dict[str, Any]:
    """SVM lineaire calibré (reference forte sur one-hot Amazon)."""
    clf = CalibratedClassifierCV(
        LinearSVC(class_weight="balanced", max_iter=3000, random_state=seed),
        cv=3,
        method="sigmoid",
    )
    x_train = np.asarray(x_train, dtype=np.float32)
    clf.fit(x_train, y_train)

    proba_val = clf.predict_proba(x_val)
    proba_test = clf.predict_proba(x_test)
    thresh, val_tune = tune_grant_threshold(proba_val, y_val, metric="deny_f1")

    pred_test = predict_with_grant_threshold(proba_test, thresh)
    pred_argmax = clf.predict(x_test)

    return {
        "model": "LinearSVM_calibrated",
        "grant_threshold": thresh,
        "val_tune": val_tune,
        "test_auroc": float(roc_auc_score(y_test, proba_test[:, 1])),
        "test_accuracy": float(accuracy_score(y_test, pred_argmax)),
        "test_deny_recall_tuned": float(recall_score(y_test, pred_test, pos_label=0, zero_division=0)),
        "test_deny_f1_tuned": float(f1_score(y_test, pred_test, pos_label=0, zero_division=0)),
        "test_balanced_accuracy_tuned": float(balanced_accuracy_score(y_test, pred_test)),
    }
