"""
Synthetic access-control dataset (tabular, binary labels).
Features are in [0,1]; labels 1=Allow, 0=Deny.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .preprocessing import AccessFeatureSpec


@dataclass
class SyntheticAccessDataset:
    """Container with train/val/test numpy arrays."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    spec: AccessFeatureSpec


def _make_features(n: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """
    Columns (interpretable toy ABAC):
      0:2 role one-hot (3) — only first 3 columns are categorical one-hot
      3: clearance [0,1]
      4:7 resource_sensitivity one-hot (4)
      8:9 action one-hot (2)
      10: emergency_override binary
    Total D = 11
    """
    role = np.zeros((n, 3), dtype=np.float32)
    ridx = rng.integers(0, 3, size=n)
    role[np.arange(n), ridx] = 1.0

    clearance = rng.random((n, 1), dtype=np.float32)

    res = np.zeros((n, 4), dtype=np.float32)
    r2 = rng.integers(0, 4, size=n)
    res[np.arange(n), r2] = 1.0

    act = np.zeros((n, 2), dtype=np.float32)
    aidx = rng.integers(0, 2, size=n)
    act[np.arange(n), aidx] = 1.0

    emergency = (rng.random((n, 1), dtype=np.float32) > 0.92).astype(np.float32)

    X = np.concatenate([role, clearance, res, act, emergency], axis=1)

    # Rule: Allow if emergency OR (clearance >= sensitivity and not destructive action)
    sens = r2.astype(np.float32) / 3.0
    cle = clearance[:, 0]
    destructive = act[:, 1]  # second action = "write/delete"
    allow = ((cle + 0.15 >= sens) & (destructive < 0.5)) | (emergency[:, 0] > 0.5)
    y = allow.astype(np.int64)
    return X, y


def default_feature_spec() -> AccessFeatureSpec:
    names = (
        [f"role_{i}" for i in range(3)]
        + ["clearance"]
        + [f"resource_{i}" for i in range(4)]
        + [f"action_{i}" for i in range(2)]
        + ["emergency"]
    )
    numerical = [3]
    categorical = list(range(0, 3)) + list(range(4, 11))
    return AccessFeatureSpec(
        numerical=numerical,
        categorical=categorical,
        feature_names=names,
    )


def load_synthetic_access_arrays(
    n_train: int = 2000,
    n_val: int = 400,
    n_test: int = 400,
    seed: int = 42,
) -> SyntheticAccessDataset:
    rng = np.random.default_rng(seed)
    X_tr, y_tr = _make_features(n_train, rng)
    X_va, y_va = _make_features(n_val, rng)
    X_te, y_te = _make_features(n_test, rng)
    return SyntheticAccessDataset(
        X_train=X_tr,
        y_train=y_tr,
        X_val=X_va,
        y_val=y_va,
        X_test=X_te,
        y_test=y_te,
        spec=default_feature_spec(),
    )


class AccessTensorDataset(Dataset):
    """Torch dataset wrapping X, y tensors."""

    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        self.X = X
        self.y = y

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]
