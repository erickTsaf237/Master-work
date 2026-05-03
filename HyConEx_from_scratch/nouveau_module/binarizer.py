from __future__ import annotations

import numpy as np


class TabularBinarizer:
    """Discrétisation en bins + encodage bipolar {-1, +1} par feature (un bin actif à +1, les autres à -1)."""

    def __init__(self, bins_per_feature: int = 4) -> None:
        self.bins_per_feature = bins_per_feature
        self.edges_: list[np.ndarray] = []
        self.feature_names_: list[str] = []
        self._offsets: list[tuple[int, int]] = []
        self.n_binary_features_: int = 0

    def fit(self, x: np.ndarray, feature_names: list[str] | None = None) -> "TabularBinarizer":
        x = np.asarray(x, dtype=np.float32)
        n_features = x.shape[1]
        self.feature_names_ = feature_names or [f"f{i}" for i in range(n_features)]

        self.edges_ = []
        self._offsets = []
        start = 0
        for j in range(n_features):
            col = x[:, j]
            qs = np.linspace(0.0, 1.0, self.bins_per_feature + 1)
            edges = np.quantile(col, qs)
            edges = np.unique(edges)
            if edges.shape[0] < 2:
                edges = np.array([col.min(), col.max() + 1e-6], dtype=np.float32)
            self.edges_.append(edges.astype(np.float32))
            n_bins = edges.shape[0] - 1
            end = start + n_bins
            self._offsets.append((start, end))
            start = end

        self.n_binary_features_ = start
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        n = x.shape[0]
        out = np.full((n, self.n_binary_features_), -1.0, dtype=np.float32)
        for j, edges in enumerate(self.edges_):
            s, e = self._offsets[j]
            bins = np.digitize(x[:, j], edges[1:-1], right=False)
            bins = np.clip(bins, 0, e - s - 1)
            out[np.arange(n), s + bins] = 1.0
        return out

    def fit_transform(self, x: np.ndarray, feature_names: list[str] | None = None) -> np.ndarray:
        return self.fit(x, feature_names=feature_names).transform(x)

    def binary_to_continuous(self, x_bin: np.ndarray) -> np.ndarray:
        x_bin = np.asarray(x_bin, dtype=np.float32)
        n = x_bin.shape[0]
        x_cont = np.zeros((n, len(self.edges_)), dtype=np.float32)

        for j, edges in enumerate(self.edges_):
            s, e = self._offsets[j]
            chunk = x_bin[:, s:e]
            # Un seul +1 par groupe ; les autres sont -1.
            idx = np.argmax(chunk, axis=1)
            left = edges[idx]
            right = edges[idx + 1]
            x_cont[:, j] = (left + right) * 0.5
        return x_cont

    def binary_feature_names(self) -> list[str]:
        names: list[str] = []
        for j, edges in enumerate(self.edges_):
            base = self.feature_names_[j]
            for b in range(edges.shape[0] - 1):
                names.append(f"{base}_bin{b}_[{edges[b]:.3f},{edges[b+1]:.3f})")
        return names
