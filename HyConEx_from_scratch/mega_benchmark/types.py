from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class UnifiedSplits:
    """Splits unifiés pour tous les jeux (DLBAC, HyConEx, HyperLogic, dataset/)."""

    dataset_id: str
    source: str
    name: str
    x_train: np.ndarray
    x_val: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    class_names: list[str]

    @property
    def num_features(self) -> int:
        return int(self.x_train.shape[1])

    @property
    def num_classes(self) -> int:
        return int(len(self.class_names))

    def safe_id(self) -> str:
        return self.dataset_id.replace("/", "__").replace(" ", "_")
