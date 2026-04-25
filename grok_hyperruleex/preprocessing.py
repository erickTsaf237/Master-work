"""Binarisation ±1 et mapping binaire → features originales (rapport §5)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


def _make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


@dataclass
class BinarizerConfig:
    numerical_cols: List[str]
    categorical_cols: List[str]


@dataclass
class TabularBinarizer:
    """
    Pipeline : ColumnTransformer (num + cat) → MinMax [0,1] sur tout le vecteur
    → passage en {-1, +1}.
    """

    config: BinarizerConfig
    _ct: ColumnTransformer | None = None
    _post: MinMaxScaler | None = None
    feature_names: List[str] = field(default_factory=list)
    binary_to_original: dict[int, str] = field(default_factory=dict)

    def fit(self, df: pd.DataFrame) -> TabularBinarizer:
        self._ct = ColumnTransformer(
            [
                (
                    "num",
                    Pipeline(
                        [
                            (
                                "mm",
                                MinMaxScaler(feature_range=(0.0, 1.0)),
                            )
                        ]
                    ),
                    self.config.numerical_cols,
                ),
                (
                    "cat",
                    _make_one_hot_encoder(),
                    self.config.categorical_cols,
                ),
            ]
        )
        Xp = self._ct.fit_transform(
            df[self.config.numerical_cols + self.config.categorical_cols]
        )
        self._post = MinMaxScaler(feature_range=(0.0, 1.0))
        self._post.fit(Xp)
        raw_names = list(self._ct.get_feature_names_out())
        self.feature_names = raw_names
        for i, rn in enumerate(raw_names):
            self.binary_to_original[i] = rn
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        assert self._ct is not None and self._post is not None
        Xp = self._ct.transform(
            df[self.config.numerical_cols + self.config.categorical_cols]
        )
        X01 = self._post.transform(Xp).astype(np.float32)
        return (2.0 * X01 - 1.0).astype(np.float32)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)
