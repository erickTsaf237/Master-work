"""Chargement et prétraitement des fichiers .sample (format DlbacAlpha)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy import loadtxt
from tensorflow.keras.utils import to_categorical

from prepare_dlbac_datasets import DLBACDatasetSpec, binary_access_label, joint_op_label


@dataclass
class DLBACAlphaArrays:
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    depth: int
    epochs: int
    num_ops: int
    hide_meta_data: int
    y_train_joint: np.ndarray
    y_test_joint: np.ndarray


def _paper_depth_and_epochs(cols: int) -> tuple[int, int]:
    hide_meta_data = cols - 20 if cols > 20 else 0
    n = 9 if hide_meta_data > 0 else 1
    depth = n * 6 + 2
    epochs = 30 if depth > 8 else 60
    return depth, epochs, hide_meta_data


def load_keras_arrays(spec: DLBACDatasetSpec) -> DLBACAlphaArrays:
    if spec.train_path is None:
        raise FileNotFoundError(f"Pas de fichier train pour {spec.name}")

    num_ops = spec.num_ops
    raw_train = loadtxt(spec.train_path, delimiter=" ", dtype=str)
    raw_test = loadtxt(spec.test_path, delimiter=" ", dtype=str)
    train_body = raw_train[:, 2:]
    test_body = raw_test[:, 2:]
    cols = train_body.shape[1]

    depth, epochs, hide_meta_data = _paper_depth_and_epochs(cols)
    metadata = cols - num_ops
    umeta_end, rmeta_end = 8, 16
    umeta_hide_end = umeta_end + hide_meta_data
    rmeta_hide_end = rmeta_end + hide_meta_data

    def _joint_labels(y: np.ndarray) -> np.ndarray:
        if spec.label_mode == "binary_access":
            return binary_access_label(y)
        return joint_op_label(y)

    def _xy(body: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = body[:, :metadata].astype(int)
        y = body[:, metadata:cols].astype(int)
        y_joint = _joint_labels(y)
        x = np.delete(x, slice(umeta_end, umeta_hide_end), axis=1)
        x = np.delete(x, slice(rmeta_end, rmeta_hide_end), axis=1)
        x = to_categorical(x)
        x = x[..., np.newaxis]
        return x, y, y_joint

    x_train, y_train, y_train_joint = _xy(train_body)
    x_test, y_test, y_test_joint = _xy(test_body)
    return DLBACAlphaArrays(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        depth=depth,
        epochs=epochs,
        num_ops=num_ops,
        hide_meta_data=hide_meta_data,
        y_train_joint=y_train_joint,
        y_test_joint=y_test_joint,
    )


def spec_from_dataset_name(name: str) -> DLBACDatasetSpec:
    from prepare_dlbac_datasets import discover_dlbac_datasets

    specs = {s.name: s for s in discover_dlbac_datasets()}
    if name not in specs:
        raise KeyError(name)
    return specs[name]
