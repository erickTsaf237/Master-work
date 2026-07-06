from __future__ import annotations

import numpy as np
import torch


def continuous_to_bipolar(x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """[0,1] ou deja bipolar -> {-1,+1}."""
    if isinstance(x, torch.Tensor):
        if x.min() >= -1.01 and x.max() <= 1.01:
            return torch.where(x > 0.25, torch.ones_like(x), -torch.ones_like(x))
        return torch.where(x > 0.5, torch.ones_like(x), -torch.ones_like(x))
    x = np.asarray(x, dtype=np.float32)
    if x.min() >= -1.01 and x.max() <= 1.01:
        return np.where(x > 0.25, 1.0, -1.0).astype(np.float32)
    return np.where(x > 0.5, 1.0, -1.0).astype(np.float32)


def bipolar_to_continuous(x_bin: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """{-1,+1} -> [0,1]."""
    if isinstance(x_bin, torch.Tensor):
        return (x_bin + 1.0) * 0.5
    return ((np.asarray(x_bin, dtype=np.float32) + 1.0) * 0.5).astype(np.float32)


def bipolar_feature_names(n: int, prefix: str = "oh") -> list[str]:
    return [f"{prefix}_{i}" for i in range(n)]
