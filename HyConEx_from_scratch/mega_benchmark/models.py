"""Entraînement / évaluation de chaque modèle sur UnifiedSplits."""

from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np
import torch
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from mega_benchmark.config import MegaBenchmarkConfig
from mega_benchmark.types import UnifiedSplits


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    if y_proba is not None and y_proba.shape[1] == 2:
        try:
            out["auc"] = float(roc_auc_score(y_true, y_proba[:, 1]))
        except ValueError:
            out["auc"] = float("nan")
    elif y_proba is not None and y_proba.shape[1] > 2:
        try:
            out["auc"] = float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro"))
        except ValueError:
            out["auc"] = float("nan")
    return out


def _batched_proba(model, x: np.ndarray, device: torch.device, bs: int = 256) -> np.ndarray:
    model.eval()
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x), bs):
            xb = torch.tensor(x[start : start + bs], dtype=torch.float32, device=device)
            chunks.append(torch.softmax(model(xb), dim=1).cpu().numpy())
    return np.vstack(chunks)


def run_sklearn_model(
    splits: UnifiedSplits,
    *,
    model_name: str,
    build_clf: Callable[[], Any],
    cfg: MegaBenchmarkConfig,
) -> dict[str, Any]:
    t0 = time.time()
    clf = build_clf()
    clf.fit(splits.x_train, splits.y_train)
    proba = clf.predict_proba(splits.x_test)
    pred = np.argmax(proba, axis=1)
    m = _metrics(splits.y_test, pred, proba)
    return {
        "model": model_name,
        "dataset_id": splits.dataset_id,
        "device": "cpu",
        "elapsed_sec": time.time() - t0,
        **m,
        "status": "ok",
    }


def run_mlp(splits: UnifiedSplits, cfg: MegaBenchmarkConfig) -> dict[str, Any]:
    if cfg.use_gpu_mlp and torch.cuda.is_available():
        return run_mlp_torch(splits, cfg)
    return run_sklearn_model(
        splits,
        model_name="MLP",
        build_clf=lambda: MLPClassifier(
            hidden_layer_sizes=(128, 64),
            max_iter=120 if not cfg.sklearn_fast else 80,
            early_stopping=True,
            random_state=cfg.seed,
        ),
        cfg=cfg,
    )


def run_mlp_torch(splits: UnifiedSplits, cfg: MegaBenchmarkConfig) -> dict[str, Any]:
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    t0 = time.time()
    device = torch.device("cuda")
    n_in = splits.num_features
    n_out = splits.num_classes
    epochs = cfg.neural_epochs_tabular

    class _MLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_in, 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, n_out),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    model = _MLP().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    ds = TensorDataset(
        torch.tensor(splits.x_train, dtype=torch.float32),
        torch.tensor(splits.y_train, dtype=torch.long),
    )
    loader = DataLoader(ds, batch_size=128, shuffle=True)
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            crit(model(xb), yb).backward()
            opt.step()

    proba = _batched_proba(model, splits.x_test, device)
    pred = proba.argmax(axis=1)
    m = _metrics(splits.y_test, pred, proba)
    return {
        "model": "MLP-GPU",
        "dataset_id": splits.dataset_id,
        "device": "cuda",
        "elapsed_sec": time.time() - t0,
        **m,
        "status": "ok",
    }


def run_rf(splits: UnifiedSplits, cfg: MegaBenchmarkConfig) -> dict[str, Any]:
    return run_sklearn_model(
        splits,
        model_name="RandomForest",
        build_clf=lambda: RandomForestClassifier(
            n_estimators=100, max_depth=14, random_state=cfg.seed, n_jobs=-1
        ),
        cfg=cfg,
    )


def run_decision_tree(splits: UnifiedSplits, cfg: MegaBenchmarkConfig) -> dict[str, Any]:
    return run_sklearn_model(
        splits,
        model_name="DecisionTree",
        build_clf=lambda: DecisionTreeClassifier(max_depth=12, random_state=cfg.seed),
        cfg=cfg,
    )


def run_svm(splits: UnifiedSplits, cfg: MegaBenchmarkConfig) -> dict[str, Any]:
    return run_sklearn_model(
        splits,
        model_name="SVM",
        build_clf=lambda: CalibratedClassifierCV(
            LinearSVC(class_weight="balanced", max_iter=3000, random_state=cfg.seed),
            cv=3,
            method="sigmoid",
        ),
        cfg=cfg,
    )


def run_ruleconex(splits: UnifiedSplits, cfg: MegaBenchmarkConfig) -> dict[str, Any]:
    from ruleconex.config import RuleConExConfig
    from ruleconex.evaluate import evaluate_counterfactuals
    from ruleconex.trainer import RuleConExTrainer

    t0 = time.time()
    is_amazon = splits.name.startswith("amazon")
    is_dlbac = splits.source == "dlbac"
    epochs = (
        cfg.neural_epochs_amazon
        if is_amazon
        else (cfg.neural_epochs_dlbac if is_dlbac else cfg.neural_epochs_tabular)
    )
    rcfg = RuleConExConfig(seed=cfg.seed, epochs=epochs, batch_size=128)
    trainer = RuleConExTrainer(rcfg, device="cuda")
    trainer.fit(
        splits.x_train,
        splits.y_train,
        splits.x_val,
        splits.y_val,
        feature_names=splits.feature_names,
        class_names=splits.class_names,
        verbose=cfg.verbose,
    )
    proba = trainer.predict_proba(splits.x_test)
    pred = proba.argmax(axis=1)
    m = _metrics(splits.y_test, pred, proba)
    cf = evaluate_counterfactuals(trainer, splits.x_test, splits.y_test, max_samples=64)
    return {
        "model": "RuleConEx",
        "dataset_id": splits.dataset_id,
        "device": "cuda",
        "elapsed_sec": time.time() - t0,
        **m,
        "cf_validity": cf.get("validity_cf"),
        "status": "ok",
    }


def run_hyconex_local(splits: UnifiedSplits, cfg: MegaBenchmarkConfig) -> dict[str, Any]:
    from hyconex_from_scratch.config import TrainConfig
    from hyconex_pure_local.trainer import HyConExLocalTrainer

    t0 = time.time()
    high = splits.num_features > 1000
    is_dlbac = splits.source == "dlbac"
    epochs = (
        cfg.neural_epochs_amazon
        if splits.name.startswith("amazon")
        else (cfg.neural_epochs_dlbac if is_dlbac else cfg.neural_epochs_tabular)
    )
    tcfg = TrainConfig(
        seed=cfg.seed,
        epochs=epochs,
        batch_size=32 if high else 128,
        latent_dim=64 if high else 32,
        hidden_dim=128 if high else 64,
    )
    trainer = HyConExLocalTrainer(tcfg, device="cuda")
    trainer.fit(splits.x_train, splits.y_train, splits.x_val, splits.y_val, verbose=False)
    proba = _batched_proba(trainer.model, splits.x_test, trainer.device)
    pred = proba.argmax(axis=1)
    m = _metrics(splits.y_test, pred, proba)
    return {
        "model": "HyConEx-Local",
        "dataset_id": splits.dataset_id,
        "device": str(trainer.device),
        "elapsed_sec": time.time() - t0,
        **m,
        "status": "ok",
    }


def run_hyperlogic(splits: UnifiedSplits, cfg: MegaBenchmarkConfig) -> dict[str, Any]:
    from hyperlogic_pure.config import PureDRConfig
    from hyperlogic_pure.trainer import PureDRNetTrainer

    t0 = time.time()
    high = splits.num_features > 512
    is_dlbac = splits.source == "dlbac"
    epochs = (
        cfg.neural_epochs_amazon
        if splits.name.startswith("amazon")
        else (cfg.neural_epochs_dlbac if is_dlbac else cfg.neural_epochs_tabular)
    )
    pcfg = PureDRConfig(seed=cfg.seed, epochs=epochs, batch_size=32 if high else 128, cf_epochs=0)
    trainer = PureDRNetTrainer(pcfg, device="cuda")
    trainer.fit(
        splits.x_train,
        splits.y_train,
        splits.x_val,
        splits.y_val,
        feature_names=splits.feature_names,
        class_names=splits.class_names,
        verbose=False,
        phase="drnet",
    )
    proba = trainer.predict_proba(splits.x_test)
    pred = proba.argmax(axis=1)
    m = _metrics(splits.y_test, pred, proba)
    return {
        "model": "HyperLogic-PureDRNet",
        "dataset_id": splits.dataset_id,
        "device": "cuda",
        "elapsed_sec": time.time() - t0,
        **m,
        "status": "ok",
    }


def run_hyconex_hyperlogic(splits: UnifiedSplits, cfg: MegaBenchmarkConfig) -> dict[str, Any]:
    from hyconex_hyperlogic import HyConExHyperLogicTrainer
    from hyconex_hyperlogic.config import HybridConfig

    t0 = time.time()
    is_amazon = splits.name.startswith("amazon")
    epochs = cfg.neural_epochs_amazon if is_amazon else cfg.neural_epochs_dlbac
    hcfg = HybridConfig(seed=cfg.seed, epochs=epochs, batch_size=128, cf_epochs=0)
    trainer = HyConExHyperLogicTrainer(hcfg, device="cuda")
    trainer.fit(
        splits.x_train,
        splits.y_train,
        splits.x_val,
        splits.y_val,
        feature_names=splits.feature_names,
        class_names=splits.class_names,
        verbose=False,
    )
    proba = trainer.predict_proba(splits.x_test)
    pred = proba.argmax(axis=1)
    m = _metrics(splits.y_test, pred, proba)
    return {
        "model": "HyConEx-HyperLogic",
        "dataset_id": splits.dataset_id,
        "device": "cuda",
        "elapsed_sec": time.time() - t0,
        **m,
        "status": "ok",
    }


def run_dlbac_alpha(splits: UnifiedSplits, cfg: MegaBenchmarkConfig) -> dict[str, Any]:
    """DLBACα ResNet officiel (Keras) sur les fichiers .sample de DlbacAlpha-main."""
    if splits.source != "dlbac":
        return {
            "model": "DLBACα-ResNet",
            "dataset_id": splits.dataset_id,
            "status": "skipped",
            "reason": "DLBACα ResNet réservé aux jeux DLBAC (fichiers .sample)",
        }

    if splits.name.startswith("amazon") or splits.num_classes <= 2:
        return {
            "model": "DLBACα-ResNet",
            "dataset_id": splits.dataset_id,
            "status": "skipped",
            "reason": "DLBACα ResNet officiel : synthétiques 4-op uniquement (pas Amazon)",
        }

    try:
        import tensorflow  # noqa: F401
    except ImportError as exc:
        return {
            "model": "DLBACα-ResNet",
            "dataset_id": splits.dataset_id,
            "status": "error",
            "error": f"TensorFlow requis (conda env hyconex) : {exc}",
        }

    from dlbac_alpha_baseline.trainer import train_eval_dlbac_alpha

    epochs = cfg.dlbac_alpha_epochs
    if epochs is None and cfg.neural_epochs_dlbac < 60:
        epochs = cfg.neural_epochs_dlbac

    row = train_eval_dlbac_alpha(splits.name, epochs=epochs, verbose=cfg.verbose)
    row["model_label"] = "DLBACα-ResNet"
    return row


def run_tabresnet_dlbac(splits: UnifiedSplits, cfg: MegaBenchmarkConfig) -> dict[str, Any]:
    if splits.source != "dlbac":
        return {
            "model": "TabResNet-DLBAC",
            "dataset_id": splits.dataset_id,
            "status": "skipped",
            "reason": "TabResNet DLBAC réservé aux jeux DLBAC one-hot",
        }

    from tabresnet_dlbac import TabResNetDLBACTrainer
    from tabresnet_dlbac.config import TabResNetDLBACConfig

    t0 = time.time()
    is_amazon = splits.name.startswith("amazon")
    epochs = cfg.neural_epochs_amazon if is_amazon else cfg.neural_epochs_dlbac
    tcfg = TabResNetDLBACConfig(seed=cfg.seed, embed_epochs=epochs, instance_epochs=min(epochs, 40))
    trainer = TabResNetDLBACTrainer(tcfg, device="cuda")
    trainer.fit(
        splits.x_train,
        splits.y_train,
        splits.x_val,
        splits.y_val,
        feature_names=splits.feature_names,
        class_names=splits.class_names,
        verbose=False,
    )
    proba = trainer.predict_proba(splits.x_test)
    pred = proba.argmax(axis=1)
    m = _metrics(splits.y_test, pred, proba)
    return {
        "model": "TabResNet-DLBAC",
        "dataset_id": splits.dataset_id,
        "device": "cuda",
        "elapsed_sec": time.time() - t0,
        **m,
        "status": "ok",
    }


MODEL_RUNNERS = {
    "ruleconex": run_ruleconex,
    "hyconex_local": run_hyconex_local,
    "hyperlogic": run_hyperlogic,
    "hyconex_hyperlogic": run_hyconex_hyperlogic,
    "dlbac_alpha": run_dlbac_alpha,
    "tabresnet_dlbac": run_tabresnet_dlbac,
    "mlp": run_mlp,
    "rf": run_rf,
    "decision_tree": run_decision_tree,
    "svm": run_svm,
}

MODEL_LABELS = {
    "ruleconex": "RuleConEx",
    "hyconex_local": "HyConEx-Local",
    "hyperlogic": "HyperLogic",
    "hyconex_hyperlogic": "HyConEx+HyperLogic",
    "dlbac_alpha": "DLBACα-ResNet",
    "tabresnet_dlbac": "TabResNet (DLBAC)",
    "mlp": "MLP",
    "rf": "RandomForest",
    "decision_tree": "DecisionTree",
    "svm": "SVM",
}
