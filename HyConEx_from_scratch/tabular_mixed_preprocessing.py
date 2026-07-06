"""Prétraitement tabulaire mixte : continus (intervalles) + catégorielles (one-hot)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from nouveau_module.binarizer import TabularBinarizer


@dataclass
class TabularDatasetSplits:
    name: str
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    class_names: list[str]
    continuous_cols: list[str]
    categorical_cols: list[str]
    input_encoding: str  # "quantize" | "bipolar"


class MixedTabularPreprocessor:
    """Continus → quantiles (intervalles actifs) ; catégorielles → one-hot ; sortie {0,1}."""

    def __init__(self, *, bins_per_feature: int = 4) -> None:
        self.bins_per_feature = bins_per_feature
        self.continuous_cols_: list[str] = []
        self.categorical_cols_: list[str] = []
        self.scaler_ = MinMaxScaler()
        self.cont_binarizer_ = TabularBinarizer(
            bins_per_feature=bins_per_feature,
            encoding="quantize",
        )
        self.cat_encoder_: OneHotEncoder | None = None
        self.feature_names_: list[str] = []

    def fit(self, df: pd.DataFrame, *, continuous_cols: list[str], categorical_cols: list[str]) -> "MixedTabularPreprocessor":
        self.continuous_cols_ = list(continuous_cols)
        self.categorical_cols_ = list(categorical_cols)

        parts: list[np.ndarray] = []
        names: list[str] = []

        if self.continuous_cols_:
            x_cont = self.scaler_.fit_transform(df[self.continuous_cols_].astype(np.float32))
            self.cont_binarizer_.fit(x_cont, feature_names=self.continuous_cols_)
            x_cont_bin = self.cont_binarizer_.transform(x_cont)
            parts.append(((x_cont_bin + 1.0) * 0.5).astype(np.float32))
            names.extend(self.cont_binarizer_.binary_feature_names())

        if self.categorical_cols_:
            self.cat_encoder_ = OneHotEncoder(
                sparse_output=False,
                handle_unknown="ignore",
            )
            x_cat = self.cat_encoder_.fit_transform(df[self.categorical_cols_].astype(str))
            parts.append(x_cat.astype(np.float32))
            for col, cats in zip(self.categorical_cols_, self.cat_encoder_.categories_):
                for cat in cats:
                    names.append(f"{col}={cat}")

        self.feature_names_ = names
        if not parts:
            raise ValueError("Au moins une colonne continue ou catégorielle est requise.")
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        parts: list[np.ndarray] = []
        if self.continuous_cols_:
            x_cont = self.scaler_.transform(df[self.continuous_cols_].astype(np.float32))
            x_cont_bin = self.cont_binarizer_.transform(x_cont)
            parts.append(((x_cont_bin + 1.0) * 0.5).astype(np.float32))
        if self.categorical_cols_ and self.cat_encoder_ is not None:
            x_cat = self.cat_encoder_.transform(df[self.categorical_cols_].astype(str))
            parts.append(x_cat.astype(np.float32))
        return np.concatenate(parts, axis=1).astype(np.float32)


def _wrap_array_splits(
    name: str,
    *,
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    class_names: list[str],
    continuous_cols: list[str],
    categorical_cols: list[str],
    input_encoding: str,
) -> TabularDatasetSplits:
    return TabularDatasetSplits(
        name=name,
        X_train=x_train.astype(np.float32),
        X_val=x_val.astype(np.float32),
        X_test=x_test.astype(np.float32),
        y_train=y_train.astype(np.int64),
        y_val=y_val.astype(np.int64),
        y_test=y_test.astype(np.int64),
        feature_names=feature_names,
        class_names=class_names,
        continuous_cols=continuous_cols,
        categorical_cols=categorical_cols,
        input_encoding=input_encoding,
    )


def get_dataset_loaders() -> dict[str, Callable[[], TabularDatasetSplits]]:
    from hyconex_hyperlogic_datasets import build_all_dataset_loaders

    return build_all_dataset_loaders()
