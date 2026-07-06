"""Évaluation RuleConEx et baselines."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.neural_network import MLPClassifier

import torch

from hyconex_from_scratch.config import TrainConfig
from hyconex_pure_local.trainer import HyConExLocalTrainer
from hyperlogic_pure.config import PureDRConfig
from hyperlogic_pure.trainer import PureDRNetTrainer
from nouveau_module.binary_metrics import tune_grant_threshold
from ruleconex.trainer import RuleConExTrainer
from ruleconex.utils import numpy_to_model_input


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None = None) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    out: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    if y_proba is not None and y_proba.shape[1] == 2:
        try:
            out["auc"] = float(roc_auc_score(y_true, y_proba[:, 1]))
        except ValueError:
            out["auc"] = float("nan")
    return out


def evaluate_counterfactuals(
    trainer: RuleConExTrainer,
    X: np.ndarray,
    y: np.ndarray,
    *,
    max_samples: int = 128,
    batch_size: int | None = None,
) -> dict[str, float]:
    """
    Métriques contrefactuelles HyConEx :
    - validity_cf : part des CF qui atteignent la classe cible (c != y)
    - changed_features_mean : nombre moyen de colonnes one-hot activées/désactivées (vrai flip 0↔1)
    - proximity_l1_mean : distance L1 moyenne ||x' - x||_1
    - flip_success_rate : part des échantillons avec au moins un CF valide
    """
    assert trainer.model is not None
    model = trainer.model
    model.eval()

    X = numpy_to_model_input(X)
    y = np.asarray(y, dtype=np.int64)
    if len(X) > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X), size=max_samples, replace=False)
        X, y = X[idx], y[idx]

    bs = batch_size or trainer.config.eval_batch_size
    num_classes = model.num_classes

    validity_hits = 0
    validity_total = 0
    changed_sum = 0.0
    l1_sum = 0.0
    sample_has_valid = 0
    n_samples = 0

    with torch.no_grad():
        for start in range(0, len(X), bs):
            xb = torch.tensor(X[start : start + bs], dtype=torch.float32, device=trainer.device)
            yb = torch.tensor(y[start : start + bs], dtype=torch.long, device=trainer.device)
            bsz = xb.shape[0]

            x_cf_all, logits_cf_all = model.generate_counterfactuals_all_classes(xb, yb)
            y_cf_pred = logits_cf_all.argmax(dim=2)

            pack = model.forward_pack(xb)
            assert pack.enc_in is not None
            x_orig = pack.enc_in

            for i in range(bsz):
                yi = int(yb[i].item())
                any_valid = False
                for c in range(num_classes):
                    if c == yi:
                        continue
                    validity_total += 1
                    pred_c = int(y_cf_pred[i, c].item())
                    if pred_c == c:
                        validity_hits += 1
                        any_valid = True

                    delta = (x_cf_all[i, c] - x_orig[i]).abs()
                    before = x_orig[i] > 0.5
                    after = x_cf_all[i, c] > 0.5
                    n_flip = int((before != after).sum().item())
                    if n_flip == 0:
                        n_flip = int((delta > 0.5).sum().item())
                    changed_sum += n_flip
                    l1_sum += float(delta.sum().item())

                if any_valid:
                    sample_has_valid += 1
                n_samples += 1

    n_cf = max(validity_total, 1)
    return {
        "validity_cf": validity_hits / n_cf,
        "changed_features_mean": changed_sum / n_cf,
        "proximity_l1_mean": l1_sum / n_cf,
        "flip_success_rate": sample_has_valid / max(n_samples, 1),
        "n_samples_evaluated": float(n_samples),
        "n_counterfactuals": float(validity_total),
    }


def evaluate_ruleconex(
    trainer: RuleConExTrainer,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    tune_threshold: bool = True,
) -> dict[str, Any]:
    proba = trainer.predict_proba(X_test)
    y_pred = proba.argmax(axis=1)
    metrics = classification_metrics(y_test, y_pred, proba)

    out: dict[str, Any] = {"metrics": metrics, "y_pred": y_pred, "y_proba": proba}
    out["counterfactuals"] = evaluate_counterfactuals(trainer, X_test, y_test)
    if tune_threshold and proba.shape[1] == 2:
        t, tuned = tune_grant_threshold(proba, y_test, metric="deny_f1")
        out["threshold"] = t
        out["tuned_metrics"] = tuned
    return out


def baseline_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    seed: int = 42,
) -> dict[str, Any]:
    clf = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        max_iter=80,
        random_state=seed,
        early_stopping=True,
    )
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)
    y_pred = clf.predict(X_test)
    return {"name": "MLP", "metrics": classification_metrics(y_test, y_pred, proba)}


def baseline_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    seed: int = 42,
) -> dict[str, Any]:
    clf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=seed, n_jobs=-1)
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)
    y_pred = clf.predict(X_test)
    return {"name": "RandomForest", "metrics": classification_metrics(y_test, y_pred, proba)}


def _trainer_predict_proba(trainer, X: np.ndarray) -> np.ndarray:
    if hasattr(trainer, "predict_proba"):
        return trainer.predict_proba(X)
    model = trainer.model
    device = trainer.device
    model.eval()
    x_t = torch.tensor(numpy_to_model_input(X), dtype=torch.float32, device=device)
    with torch.no_grad():
        logits = model(x_t)
        return torch.softmax(logits, dim=1).cpu().numpy()


def baseline_hyconex_local(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    epochs: int = 25,
    seed: int = 42,
) -> dict[str, Any]:
    cfg = TrainConfig(seed=seed, epochs=epochs, batch_size=128, latent_dim=64, hidden_dim=128)
    tr = HyConExLocalTrainer(cfg)
    tr.fit(X_train, y_train, X_val, y_val, verbose=False)
    proba = _trainer_predict_proba(tr, X_test)
    y_pred = proba.argmax(axis=1)
    return {"name": "HyConEx-Local", "metrics": classification_metrics(y_test, y_pred, proba)}


def baseline_hyperlogic(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    epochs: int = 25,
    seed: int = 42,
) -> dict[str, Any]:
    cfg = PureDRConfig(seed=seed, epochs=epochs, batch_size=128, cf_epochs=0)
    tr = PureDRNetTrainer(cfg)
    tr.fit(X_train, y_train, X_val, y_val, verbose=False, phase="drnet")
    proba = tr.predict_proba(X_test)
    y_pred = proba.argmax(axis=1)
    return {"name": "HyperLogic (PureDRNet)", "metrics": classification_metrics(y_test, y_pred, proba)}


def run_all_baselines(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    include_neural: bool = True,
    epochs: int = 25,
    seed: int = 42,
) -> list[dict[str, Any]]:
    X_train = numpy_to_model_input(X_train)
    X_val = numpy_to_model_input(X_val)
    X_test = numpy_to_model_input(X_test)

    results = [
        baseline_mlp(X_train, y_train, X_test, y_test, seed=seed),
        baseline_random_forest(X_train, y_train, X_test, y_test, seed=seed),
    ]
    if include_neural:
        try:
            results.append(
                baseline_hyconex_local(X_train, y_train, X_val, y_val, X_test, y_test, epochs=epochs, seed=seed)
            )
        except Exception as exc:
            results.append({"name": "HyConEx-Local", "error": str(exc)})
        try:
            results.append(
                baseline_hyperlogic(X_train, y_train, X_val, y_val, X_test, y_test, epochs=epochs, seed=seed)
            )
        except Exception as exc:
            results.append({"name": "HyperLogic (PureDRNet)", "error": str(exc)})
    return results
