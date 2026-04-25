from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


NUMERIC_FEATURES = [
    "Area",
    "Perimeter",
    "MajorAxisLength",
    "MinorAxisLength",
    "AspectRation",
    "Eccentricity",
    "ConvexArea",
    "EquivDiameter",
    "Extent",
    "Solidity",
    "roundness",
    "Compactness",
    "ShapeFactor1",
    "ShapeFactor2",
    "ShapeFactor3",
    "ShapeFactor4",
]


@dataclass
class DryBeanSplits:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    class_names: list[str]
    raw_df: pd.DataFrame


def _parse_arff_to_dataframe(arff_path: Path) -> pd.DataFrame:
    rows: list[list[str]] = []
    reading_data = False
    with arff_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("%"):
                continue
            if not reading_data:
                if line.lower() == "@data":
                    reading_data = True
                continue
            rows.append([x.strip() for x in line.split(",")])

    cols = NUMERIC_FEATURES + ["Class"]
    df = pd.DataFrame(rows, columns=cols)
    for c in NUMERIC_FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna().reset_index(drop=True)
    return df


def _build_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    eps = 1e-6

    out["Area_over_Perimeter"] = out["Area"] / (out["Perimeter"] + eps)
    out["ConvexArea_over_Area"] = out["ConvexArea"] / (out["Area"] + eps)
    out["Major_over_Minor"] = out["MajorAxisLength"] / (out["MinorAxisLength"] + eps)
    out["Roundness_x_Compactness"] = out["roundness"] * out["Compactness"]
    out["SF_sum"] = out["ShapeFactor1"] + out["ShapeFactor2"] + out["ShapeFactor3"] + out["ShapeFactor4"]

    # Log transforms for strongly skewed geometric measures.
    out["log_Area"] = np.log1p(out["Area"])
    out["log_Perimeter"] = np.log1p(out["Perimeter"])
    out["log_ConvexArea"] = np.log1p(out["ConvexArea"])

    return out


def load_dry_bean_dataframe(data_path: str | Path | None = None) -> pd.DataFrame:
    if data_path is None:
        repo_root = Path(__file__).resolve().parents[1]
        data_path = repo_root / "dataset" / "dry+bean+dataset" / "DryBeanDataset" / "Dry_Bean_Dataset.arff"
    data_path = Path(data_path)
    return _parse_arff_to_dataframe(data_path)


def prepare_dry_bean_splits(
    data_path: str | Path | None = None,
    *,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
    add_engineered_features: bool = True,
    clip_outliers: bool = True,
) -> DryBeanSplits:
    df = load_dry_bean_dataframe(data_path)

    class_names = sorted(df["Class"].unique().tolist())
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    y = df["Class"].map(class_to_idx).to_numpy(dtype=np.int64)

    X_df = df.drop(columns=["Class"]).copy()
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

    return DryBeanSplits(
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
