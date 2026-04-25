"""Tabular preprocessing: ±1 tensor for HyperLogic DR-Net, floats for HyConEx."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch


@dataclass
class AccessFeatureSpec:
    """Column indices after a fixed feature order."""

    numerical: List[int]
    categorical: List[int]
    feature_names: List[str]


def binary_rows_to_pm_one(x01: np.ndarray) -> np.ndarray:
    """Map {0,1} (or float in [0,1]) to {-1,+1}."""
    return (2.0 * np.asarray(x01, dtype=np.float32).clip(0.0, 1.0) - 1.0).astype(
        np.float32
    )


def pm_one_to_unit_interval(xpm: torch.Tensor) -> torch.Tensor:
    """Map {-1,+1} to [0,1] for HyConEx-style continuous inputs."""
    return (xpm + 1.0) * 0.5


def minmax_fit_transform(
    x: np.ndarray, numerical_idx: List[int]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return transformed copy, mins, maxs for numerical columns only."""
    out = np.array(x, dtype=np.float32, copy=True)
    mins = np.zeros(x.shape[1], dtype=np.float32)
    maxs = np.ones(x.shape[1], dtype=np.float32)
    for j in numerical_idx:
        col = out[:, j]
        mins[j] = float(np.nanmin(col))
        maxs[j] = float(np.nanmax(col))
        denom = max(maxs[j] - mins[j], 1e-6)
        out[:, j] = (col - mins[j]) / denom
    return out, mins, maxs


def minmax_transform(
    x: np.ndarray, numerical_idx: List[int], mins: np.ndarray, maxs: np.ndarray
) -> np.ndarray:
    out = np.array(x, dtype=np.float32, copy=True)
    for j in numerical_idx:
        denom = max(maxs[j] - mins[j], 1e-6)
        out[:, j] = (out[:, j] - mins[j]) / denom
    return out


class AccessControlPreprocessor:
    """
    Fits on training data in [0,1] / binary form (wide table).
    Produces:
      - x_hyperlogic: (N, D) float32 in {-1,+1} for all binary/binned dims
      - x_hyconex: (N, D) float32 in [0,1] (same width; categorical already one-hot)
    """

    def __init__(self, spec: AccessFeatureSpec):
        self.spec = spec
        self._mins: np.ndarray | None = None
        self._maxs: np.ndarray | None = None

    def fit(self, x_train: np.ndarray) -> "AccessControlPreprocessor":
        xh, mins, maxs = minmax_fit_transform(
            x_train, self.spec.numerical
        )
        self._mins, self._maxs = mins, maxs
        _ = xh
        return self

    def transform_hyperlogic(self, x: np.ndarray) -> torch.Tensor:
        if self._mins is None:
            raise RuntimeError("Call fit() before transform.")
        xn = minmax_transform(
            x, self.spec.numerical, self._mins, self._maxs
        )
        return torch.from_numpy(binary_rows_to_pm_one(xn))

    def transform_hyconex(self, x: np.ndarray) -> torch.Tensor:
        if self._mins is None:
            raise RuntimeError("Call fit() before transform.")
        xn = minmax_transform(
            x, self.spec.numerical, self._mins, self._maxs
        )
        return torch.from_numpy(np.clip(xn, 0.0, 1.0).astype(np.float32))
