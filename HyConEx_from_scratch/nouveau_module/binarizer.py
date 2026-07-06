from __future__ import annotations

from typing import Literal

import numpy as np

EncodingMode = Literal["auto", "bipolar", "quantize"]


class TabularBinarizer:
    """Prétraitement tabulaire vers entrées bipolar {-1, +1}.

    - ``bipolar`` / ``auto`` sur données déjà binaires {0,1} ou {-1,+1} : pas de quantiles,
      mapping direct 0→-1, 1→+1 (one-hot DLBAC inclus).
    - ``quantize`` : discrétisation par quantiles puis un littéral actif par variable (Iris, etc.).
    """

    def __init__(
        self,
        bins_per_feature: int = 4,
        *,
        encoding: EncodingMode = "auto",
    ) -> None:
        self.bins_per_feature = bins_per_feature
        self.encoding: EncodingMode = encoding
        self.mode_: Literal["bipolar", "quantize"] = "quantize"
        self.edges_: list[np.ndarray] = []
        self.feature_names_: list[str] = []
        self._offsets: list[tuple[int, int]] = []
        self.n_binary_features_: int = 0

    @staticmethod
    def _looks_already_binary(x: np.ndarray, atol: float = 1e-4) -> bool:
        x = np.asarray(x, dtype=np.float32)
        if x.size == 0:
            return True
        flat = x.reshape(-1)
        mn, mx = float(flat.min()), float(flat.max())
        if mn >= -1.0 - atol and mx <= 1.0 + atol:
            near = (
                np.isclose(flat, -1.0, atol=atol)
                | np.isclose(flat, 1.0, atol=atol)
                | np.isclose(flat, 0.0, atol=atol)
            )
            if float(near.mean()) > 0.995:
                return True
        if mn >= -atol and mx <= 1.0 + atol:
            near01 = np.isclose(flat, 0.0, atol=atol) | np.isclose(flat, 1.0, atol=atol)
            if float(near01.mean()) > 0.995:
                return True
        return False

    @staticmethod
    def to_bipolar(x: np.ndarray) -> np.ndarray:
        """{0,1} ou [-1,1] → {-1,+1}, sans redimensionner."""
        x = np.asarray(x, dtype=np.float32)
        flat = x.reshape(-1)
        atol = 1e-4
        # One-hot DLBAC et binaire non signe : 0→-1, 1→+1
        if flat.min() >= -atol and flat.max() <= 1.0 + atol:
            if float((np.isclose(flat, 0.0, atol=atol) | np.isclose(flat, 1.0, atol=atol)).mean()) > 0.995:
                return np.where(x > 0.5, 1.0, -1.0).astype(np.float32)
        # Deja en {-1,+1}
        if flat.min() >= -1.0 - atol and flat.max() <= 1.0 + atol:
            return np.where(x > 0.0, 1.0, -1.0).astype(np.float32)
        return np.where(x > 0.5, 1.0, -1.0).astype(np.float32)

    def fit(self, x: np.ndarray, feature_names: list[str] | None = None) -> "TabularBinarizer":
        x = np.asarray(x, dtype=np.float32)
        n_features = x.shape[1]
        self.feature_names_ = feature_names or [f"f{i}" for i in range(n_features)]

        if self.encoding == "bipolar" or (
            self.encoding == "auto" and self._looks_already_binary(x)
        ):
            self.mode_ = "bipolar"
            self.edges_ = []
            self._offsets = []
            self.n_binary_features_ = n_features
            return self

        self.mode_ = "quantize"
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
        if self.mode_ == "bipolar":
            if x.shape[1] != self.n_binary_features_:
                raise ValueError(
                    f"Dimension attendue {self.n_binary_features_}, reçue {x.shape[1]}"
                )
            return self.to_bipolar(x)

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
        if self.mode_ == "bipolar":
            return ((x_bin + 1.0) * 0.5).astype(np.float32)

        n = x_bin.shape[0]
        x_cont = np.zeros((n, len(self.edges_)), dtype=np.float32)
        for j, edges in enumerate(self.edges_):
            s, e = self._offsets[j]
            chunk = x_bin[:, s:e]
            idx = np.argmax(chunk, axis=1)
            left = edges[idx]
            right = edges[idx + 1]
            x_cont[:, j] = (left + right) * 0.5
        return x_cont

    def binary_feature_names(self) -> list[str]:
        if self.mode_ == "bipolar":
            return list(self.feature_names_)
        names: list[str] = []
        for j, edges in enumerate(self.edges_):
            base = self.feature_names_[j]
            for b in range(edges.shape[0] - 1):
                names.append(f"{base}_bin{b}_[{edges[b]:.3f},{edges[b + 1]:.3f})")
        return names
