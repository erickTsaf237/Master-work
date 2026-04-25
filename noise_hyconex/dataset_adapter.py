"""Description des colonnes encodées pour pertes CF (indices numériques / catégoriels)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


def _make_ohe() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


@dataclass
class TabularFeatureLayout:
    """Aligné sur l'usage HyConEx de numerical_features / categorical_features (indices dans x)."""

    numerical_features: List[int]
    categorical_features: List[int]
    cat_group_slices: List[Tuple[int, int]]
    feature_transformer: ColumnTransformer

    @property
    def dim(self) -> int:
        return len(self.numerical_features) + len(self.categorical_features)


def build_adult_layout(
    numerical_cols: Sequence[str],
    categorical_cols: Sequence[str],
) -> Tuple[TabularFeatureLayout, ColumnTransformer]:
    """Construit un ColumnTransformer + layout à partir des noms de colonnes (fit ultérieur)."""
    ct = ColumnTransformer(
        [
            ("num", Pipeline([("mm", MinMaxScaler(feature_range=(0.0, 1.0)))]), list(numerical_cols)),
            ("cat", _make_ohe(), list(categorical_cols)),
        ]
    )
    n_num = len(numerical_cols)
    return TabularFeatureLayout(
        numerical_features=list(range(n_num)),
        categorical_features=[],
        cat_group_slices=[],
        feature_transformer=ct,
    ), ct


def fit_layout_from_df(
    df: pd.DataFrame,
    numerical_cols: Sequence[str],
    categorical_cols: Sequence[str],
) -> TabularFeatureLayout:
    """Fit le ColumnTransformer et remplit numerical_features, categorical_features, cat_group_slices."""
    layout_base, ct = build_adult_layout(numerical_cols, categorical_cols)
    ct.fit(df[list(numerical_cols) + list(categorical_cols)])
    raw_names = list(ct.get_feature_names_out())
    n_num = len(numerical_cols)
    numerical_features = list(range(n_num))
    categorical_features = list(range(n_num, len(raw_names)))
    ohe: OneHotEncoder = ct.named_transformers_["cat"]
    cat_slices: List[Tuple[int, int]] = []
    pos = n_num
    for n_cat in [len(c) for c in ohe.categories_]:
        cat_slices.append((pos, n_cat))
        pos += n_cat
    return TabularFeatureLayout(
        numerical_features=numerical_features,
        categorical_features=categorical_features,
        cat_group_slices=cat_slices,
        feature_transformer=ct,
    )
