"""
Préparation Amazon (DLBAC) sur le même principe que Dry Bean :

- métadonnées brutes (entiers) après masque DLBAC ;
- features dérivées (ratios, logs) ;
- clipping des extrêmes (quantiles 1 % / 99 %) ;
- normalisation MinMax (fit sur train) ;
- pas de OneHotEncoder (le modèle binarise par quantiles, mode ``quantize``).

Usage :
    from feature_engineering_amazon import prepare_amazon_bean_splits
    splits = prepare_amazon_bean_splits(spec)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from prepare_dlbac_datasets import (
    DLBACDatasetSpec,
    class_names_for_labels,
)
from train_nouveau_module_dlbac_quantile import (
    DlbacQuantileSplits,
    load_dlbac_raw_arrays,
)


@dataclass
class AmazonBeanSplits:
    x_train: np.ndarray
    x_val: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    class_names: list[str]
    scaler: MinMaxScaler


def _meta_column_names(n_meta: int) -> list[str]:
    if n_meta <= 8:
        names = [f"umeta_{i}" for i in range(min(8, n_meta))]
        if n_meta > 8:
            names.append("rmeta_0")
        return names[:n_meta]
    names = [f"umeta_{i}" for i in range(8)]
    names.extend(f"rmeta_{i}" for i in range(n_meta - 8))
    return names


def _build_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ratios / logs sur les métadonnées (analogue à feature_engineering_dry_bean)."""
    out = df.copy()
    eps = 1e-6
    cols = list(df.columns)
    n = len(cols)

    for c in cols:
        out[f"log1p_{c}"] = np.log1p(out[c].clip(lower=0))

    if n >= 2:
        out[f"{cols[0]}_over_{cols[-1]}"] = out[cols[0]] / (out[cols[-1]] + eps)
        out["meta_sum"] = out[cols].sum(axis=1)
        out["meta_mean"] = out[cols].mean(axis=1)
        out["meta_std"] = out[cols].std(axis=1).fillna(0.0)

    if n >= 8:
        user_cols = cols[:8] if n > 8 else cols[:-1]
        if user_cols:
            out["user_meta_sum"] = out[user_cols].sum(axis=1)
            out["user_meta_mean"] = out[user_cols].mean(axis=1)

    return out


def prepare_amazon_bean_splits(
    spec: DLBACDatasetSpec,
    *,
    val_size: float = 0.2,
    random_state: int = 42,
    add_engineered_features: bool = True,
    clip_outliers: bool = True,
) -> AmazonBeanSplits:
    if not spec.has_train:
        raise ValueError(f"Pas de fichier train pour {spec.name}")

    x_tr, y_tr, x_te, y_te = load_dlbac_raw_arrays(
        spec.train_path,  # type: ignore[arg-type]
        spec.test_path,
        num_ops=spec.num_ops,
        label_mode=spec.label_mode,
    )

    base_names = _meta_column_names(x_tr.shape[1])
    df_tr = pd.DataFrame(x_tr, columns=base_names)
    df_te = pd.DataFrame(x_te, columns=base_names)

    if add_engineered_features:
        df_tr = _build_engineered_features(df_tr)
        df_te = _build_engineered_features(df_te)

    if clip_outliers:
        q_low = df_tr.quantile(0.01)
        q_high = df_tr.quantile(0.99)
        df_tr = df_tr.clip(lower=q_low, upper=q_high, axis=1)
        df_te = df_te.clip(lower=q_low, upper=q_high, axis=1)

    x_train_full, x_test, y_train_full, y_test = df_tr, df_te, y_tr, y_te
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full,
        y_train_full,
        test_size=val_size,
        random_state=random_state,
        stratify=y_train_full,
    )

    scaler = MinMaxScaler()
    x_train_s = scaler.fit_transform(x_train).astype(np.float32)
    x_val_s = scaler.transform(x_val).astype(np.float32)
    x_test_s = scaler.transform(x_test).astype(np.float32)

    class_names = class_names_for_labels(
        np.concatenate([y_train, y_test]),
        spec.label_mode,  # type: ignore[arg-type]
    )

    return AmazonBeanSplits(
        x_train=x_train_s,
        x_val=x_val_s,
        x_test=x_test_s,
        y_train=y_train.astype(np.int64),
        y_val=y_val.astype(np.int64),
        y_test=y_test.astype(np.int64),
        feature_names=df_tr.columns.tolist(),
        class_names=class_names,
        scaler=scaler,
    )


def build_bean_style_splits(
    spec: DLBACDatasetSpec,
    *,
    val_size: float = 0.2,
    random_state: int = 42,
    add_engineered_features: bool = True,
    clip_outliers: bool = True,
) -> DlbacQuantileSplits:
    """Wrapper compatible avec le pipeline d'entraînement Amazon."""
    bean = prepare_amazon_bean_splits(
        spec,
        val_size=val_size,
        random_state=random_state,
        add_engineered_features=add_engineered_features,
        clip_outliers=clip_outliers,
    )
    num_classes = int(max(bean.y_train.max(), bean.y_test.max()) + 1)
    return DlbacQuantileSplits(
        name=spec.name,
        kind=spec.kind,
        label_mode=spec.label_mode,
        x_train=bean.x_train,
        y_train=bean.y_train,
        x_val=bean.x_val,
        y_val=bean.y_val,
        x_test=bean.x_test,
        y_test=bean.y_test,
        feature_names=bean.feature_names,
        class_names=bean.class_names,
        num_classes=num_classes,
        scaler=bean.scaler,
        input_encoding="bean",
        onehot_dim_full=None,
    )
