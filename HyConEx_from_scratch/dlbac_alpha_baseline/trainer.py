"""Entraîne et évalue DLBACα ResNet sur les données brutes DlbacAlpha-main."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from dlbac_alpha_baseline.preprocess import DLBACAlphaArrays, load_keras_arrays, spec_from_dataset_name
from dlbac_alpha_baseline.resnet import resnet_v1
from prepare_dlbac_datasets import DLBACDatasetSpec, joint_op_label


def _configure_tf() -> None:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    try:
        import tensorflow as tf

        gpus = tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass


def _lr_schedule_resnet8(epoch: int) -> float:
    lr = 1e-3
    if epoch > 59:
        lr *= 1e-3
    elif epoch > 39:
        lr *= 1e-2
    elif epoch > 19:
        lr *= 1e-1
    return lr


def _lr_schedule_resnet56(epoch: int) -> float:
    lr = 1e-3
    if epoch > 29:
        lr *= 1e-3
    elif epoch > 19:
        lr *= 1e-2
    elif epoch > 9:
        lr *= 1e-1
    return lr


def _ops_to_joint(y_pred_ops: np.ndarray) -> np.ndarray:
    return joint_op_label((y_pred_ops > 0.5).astype(np.int64))


def _joint_predictions(y_prob: np.ndarray, label_mode: str) -> np.ndarray:
    if label_mode == "binary_access":
        col = y_prob[:, 0] if y_prob.shape[1] == 1 else y_prob.max(axis=1)
        return (col > 0.5).astype(np.int64)
    return _ops_to_joint(y_prob)


def _device_label() -> str:
    try:
        import tensorflow as tf

        return "cuda" if tf.config.list_physical_devices("GPU") else "cpu"
    except Exception:
        return "cpu"


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _build_config(
    spec: DLBACDatasetSpec,
    arrays: DLBACAlphaArrays,
    *,
    epochs: int,
    batch_size: int,
) -> dict[str, Any]:
    return {
        "model": "DLBACα-ResNet",
        "dataset": spec.name,
        "kind": spec.kind,
        "label_mode": spec.label_mode,
        "num_ops": arrays.num_ops,
        "resnet_depth": arrays.depth,
        "hide_meta_data": arrays.hide_meta_data,
        "epochs": epochs,
        "batch_size": batch_size,
        "loss": "binary_crossentropy",
        "optimizer": "Adam",
        "lr_schedule": "resnet56" if arrays.depth > 8 else "resnet8",
        "input_shape": list(arrays.x_train.shape[1:]),
        "n_train": int(arrays.x_train.shape[0]),
        "n_test": int(arrays.x_test.shape[0]),
        "num_joint_classes": int(np.unique(arrays.y_test_joint).size),
        "train_path": str(spec.train_path) if spec.train_path else None,
        "test_path": str(spec.test_path),
    }


def _compute_metrics(
    arrays: DLBACAlphaArrays,
    spec: DLBACDatasetSpec,
    y_prob: np.ndarray,
    keras_scores: list[float],
) -> dict[str, Any]:
    y_pred_joint = _joint_predictions(y_prob, spec.label_mode)
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(arrays.y_test_joint, y_pred_joint)),
        "f1_macro": float(
            f1_score(arrays.y_test_joint, y_pred_joint, average="macro", zero_division=0)
        ),
        "f1_weighted": float(
            f1_score(arrays.y_test_joint, y_pred_joint, average="weighted", zero_division=0)
        ),
        "binary_accuracy": float(keras_scores[1]),
        "test_loss": float(keras_scores[0]),
    }

    if spec.label_mode == "binary_access" or arrays.num_ops == 1:
        try:
            y_bin = arrays.y_test_joint
            proba_bin = y_prob[:, 0] if y_prob.shape[1] == 1 else y_prob.max(axis=1)
            metrics["auc"] = float(roc_auc_score(y_bin, proba_bin))
        except ValueError:
            metrics["auc"] = float("nan")
    elif y_prob.shape[1] > 2:
        try:
            metrics["auc"] = float(
                roc_auc_score(
                    arrays.y_test_joint,
                    y_prob,
                    multi_class="ovr",
                    average="macro",
                )
            )
        except ValueError:
            metrics["auc"] = float("nan")

    return metrics


def _save_artifacts(
    out_dir: Path,
    model,
    config: dict[str, Any],
    history: dict[str, list],
    metrics: dict[str, Any],
) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "model.keras"
    config_path = out_dir / "config.json"
    history_path = out_dir / "history.json"
    metrics_path = out_dir / "metrics.json"

    model.save(model_path)
    config_path.write_text(json.dumps(_json_safe(config), indent=2), encoding="utf-8")
    history_path.write_text(json.dumps(_json_safe(history), indent=2), encoding="utf-8")
    metrics_path.write_text(json.dumps(_json_safe(metrics), indent=2), encoding="utf-8")

    return {
        "model_path": str(model_path),
        "config_path": str(config_path),
        "history_path": str(history_path),
        "metrics_path": str(metrics_path),
    }


def train_eval_dlbac_alpha(
    spec: DLBACDatasetSpec | str,
    *,
    epochs: int | None = None,
    verbose: bool = False,
    out_dir: Path | str | None = None,
) -> dict[str, Any]:
    """
    Entraîne DLBACα sur train/test officiels (.sample) et évalue sur le test.

    Si ``out_dir`` est fourni, sauvegarde :
    - ``model.keras`` (poids + architecture)
    - ``config.json`` (hyperparamètres)
    - ``history.json`` (courbes d'entraînement)
    - ``metrics.json`` (métriques test)
    """
    _configure_tf()

    if isinstance(spec, str):
        spec = spec_from_dataset_name(spec)

    arrays = load_keras_arrays(spec)
    use_epochs = epochs if epochs is not None else arrays.epochs
    batch_size = int(min(max(1, arrays.x_train.shape[0] // 10), 16))

    model = resnet_v1(
        input_shape=arrays.x_train.shape[1:],
        depth=arrays.depth,
        num_classes=arrays.num_ops,
    )
    lr_fn = _lr_schedule_resnet56 if arrays.depth > 8 else _lr_schedule_resnet8
    model.compile(
        loss="binary_crossentropy",
        optimizer=Adam(learning_rate=lr_fn(0)),
        metrics=["binary_accuracy"],
    )
    callbacks = [
        ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6),
        LearningRateScheduler(lr_fn),
    ]

    config = _build_config(spec, arrays, epochs=use_epochs, batch_size=batch_size)

    t0 = time.time()
    history = model.fit(
        arrays.x_train,
        arrays.y_train,
        batch_size=batch_size,
        epochs=use_epochs,
        validation_data=(arrays.x_test, arrays.y_test),
        shuffle=True,
        callbacks=callbacks,
        verbose=1 if verbose else 0,
    )

    scores = model.evaluate(arrays.x_test, arrays.y_test, verbose=0)
    y_prob = model.predict(arrays.x_test, verbose=0)
    metrics = _compute_metrics(arrays, spec, y_prob, scores)
    metrics["n_params"] = int(model.count_params())

    out: dict[str, Any] = {
        "model": "DLBACα-ResNet",
        "dataset_id": f"dlbac/{spec.name}",
        "dataset": spec.name,
        "device": _device_label(),
        "elapsed_sec": time.time() - t0,
        "resnet_depth": arrays.depth,
        "epochs": use_epochs,
        "batch_size": batch_size,
        "status": "ok",
        **metrics,
    }

    if out_dir is not None:
        paths = _save_artifacts(
            Path(out_dir),
            model,
            config,
            history.history,
            metrics,
        )
        out.update(paths)

    return out
