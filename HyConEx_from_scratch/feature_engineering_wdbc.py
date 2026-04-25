from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


BASE_COLS = [
    "radius",
    "texture",
    "perimeter",
    "area",
    "smoothness",
    "compactness",
    "concavity",
    "concave_points",
    "symmetry",
    "fractal_dimension",
]

STATS = ["mean", "se", "worst"]

RAW_COLUMNS = ["id", "diagnosis"] + [f"{b}_{s}" for s in STATS for b in BASE_COLS]


@dataclass
class WDBCSplits:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    class_names: list[str]
    raw_df: pd.DataFrame


def _build_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    eps = 1e-6

    # Ratios "worst / mean" pour capturer l'aggravation maximale vs moyenne.
    for base in BASE_COLS:
        out[f"{base}_worst_over_mean"] = out[f"{base}_worst"] / (out[f"{base}_mean"] + eps)

    # Dispersion relative de la variabilité locale.
    for base in BASE_COLS:
        out[f"{base}_se_over_mean"] = out[f"{base}_se"] / (out[f"{base}_mean"] + eps)

    # Interactions pertinentes souvent utilisées pour WDBC.
    out["area_texture_interaction"] = out["area_mean"] * out["texture_mean"]
    out["radius_perimeter_interaction"] = out["radius_mean"] * out["perimeter_mean"]
    out["concavity_compactness_interaction"] = out["concavity_mean"] * out["compactness_mean"]

    return out


def load_wdbc_dataframe(
    data_path: str | Path | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    if data_path is None:
        repo_root = Path(__file__).resolve().parents[1]
        data_path = repo_root / "dataset" / "breast+cancer+wisconsin+diagnostic" / "wdbc.data"
    data_path = Path(data_path)

    df = pd.read_csv(data_path, header=None, names=RAW_COLUMNS)
    class_names = ["benign", "malignant"]
    df["target"] = df["diagnosis"].map({"B": 0, "M": 1}).astype(np.int64)
    df = df.drop(columns=["id", "diagnosis"])
    return df, class_names


def prepare_wdbc_splits(
    data_path: str | Path | None = None,
    *,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
    add_engineered_features: bool = True,
    clip_outliers: bool = True,
) -> WDBCSplits:
    """
    Charge WDBC, applique le feature engineering, puis fait un split train/val/test
    stratifié avec mise à l'échelle MinMax fit sur le train.
    """
    df, class_names = load_wdbc_dataframe(data_path=data_path)

    y = df["target"].to_numpy(dtype=np.int64)
    X_df = df.drop(columns=["target"]).copy()

    if add_engineered_features:
        X_df = _build_engineered_features(X_df)

    if clip_outliers:
        q_low = X_df.quantile(0.01)
        q_high = X_df.quantile(0.99)
        X_df = X_df.clip(lower=q_low, upper=q_high, axis=1)

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X_df,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=val_size,
        random_state=random_state,
        stratify=y_train_full,
    )

    scaler = MinMaxScaler()
    X_train_s = scaler.fit_transform(X_train).astype(np.float32)
    X_val_s = scaler.transform(X_val).astype(np.float32)
    X_test_s = scaler.transform(X_test).astype(np.float32)

    return WDBCSplits(
        X_train=X_train_s,
        X_val=X_val_s,
        X_test=X_test_s,
        y_train=np.asarray(y_train, dtype=np.int64),
        y_val=np.asarray(y_val, dtype=np.int64),
        y_test=np.asarray(y_test, dtype=np.int64),
        feature_names=X_df.columns.tolist(),
        class_names=class_names,
        raw_df=df,
    )
