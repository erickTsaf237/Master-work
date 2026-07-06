"""Registre unifié de tous les jeux de données."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from mega_benchmark.types import UnifiedSplits
from prepare_dlbac_datasets import discover_dlbac_datasets
from train_nouveau_module_dlbac_quantile import build_onehot_splits

ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = ROOT.parent
DATASET_ROOT = PROJECT_ROOT / "dataset"


def _to_unified(
    dataset_id: str,
    source: str,
    name: str,
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    class_names: list[str],
) -> UnifiedSplits:
    return UnifiedSplits(
        dataset_id=dataset_id,
        source=source,
        name=name,
        x_train=np.asarray(x_train, dtype=np.float32),
        x_val=np.asarray(x_val, dtype=np.float32),
        x_test=np.asarray(x_test, dtype=np.float32),
        y_train=np.asarray(y_train, dtype=np.int64),
        y_val=np.asarray(y_val, dtype=np.int64),
        y_test=np.asarray(y_test, dtype=np.int64),
        feature_names=feature_names,
        class_names=class_names,
    )


def load_dlbac_splits(spec, *, seed: int = 42) -> UnifiedSplits:
    s = build_onehot_splits(spec, val_size=0.2, random_state=seed, use_cache=True)
    return _to_unified(
        f"dlbac/{spec.name}",
        "dlbac",
        spec.name,
        s.x_train,
        s.x_val,
        s.x_test,
        s.y_train,
        s.y_val,
        s.y_test,
        s.feature_names,
        s.class_names,
    )


def load_hyconex_hyperlogic_splits(dataset_id: str, *, seed: int = 42) -> UnifiedSplits:
    from hyconex_hyperlogic_datasets import load_hyconex_tabular, load_hyperlogic_tabular

    if dataset_id.startswith("hyconex/"):
        name = dataset_id.split("/", 1)[1]
        s = load_hyconex_tabular(name, random_state=seed)
    elif dataset_id.startswith("hyperlogic/"):
        name = dataset_id.split("/", 1)[1]
        s = load_hyperlogic_tabular(name, random_state=seed)
    else:
        raise KeyError(dataset_id)
    return _to_unified(
        dataset_id,
        dataset_id.split("/")[0],
        dataset_id.split("/", 1)[1],
        s.X_train,
        s.X_val,
        s.X_test,
        s.y_train,
        s.y_val,
        s.y_test,
        s.feature_names,
        s.class_names,
    )


def load_dry_bean_splits(*, seed: int = 42) -> UnifiedSplits:
    from feature_engineering_dry_bean import prepare_dry_bean_splits

    arff = DATASET_ROOT / "dry+bean+dataset" / "DryBeanDataset" / "Dry_Bean_Dataset.arff"
    s = prepare_dry_bean_splits(arff, random_state=seed)
    return _to_unified(
        "local/dry_bean",
        "local",
        "dry_bean",
        s.X_train,
        s.X_val,
        s.X_test,
        s.y_train,
        s.y_val,
        s.y_test,
        s.feature_names,
        s.class_names,
    )


def load_wdbc_splits(*, seed: int = 42) -> UnifiedSplits:
    from feature_engineering_wdbc import prepare_wdbc_splits

    path = DATASET_ROOT / "breast+cancer+wisconsin+diagnostic" / "wdbc.data"
    s = prepare_wdbc_splits(path, random_state=seed)
    return _to_unified(
        "local/wdbc",
        "local",
        "wdbc",
        s.X_train,
        s.X_val,
        s.X_test,
        s.y_train,
        s.y_val,
        s.y_test,
        s.feature_names,
        s.class_names,
    )


def load_banknote_splits(*, seed: int = 42) -> UnifiedSplits:
    path = DATASET_ROOT / "banknote+authentication" / "data_banknote_authentication.txt"
    df = pd.read_csv(path, header=None)
    x = df.iloc[:, :-1].values.astype(np.float32)
    y = df.iloc[:, -1].values.astype(np.int64)
    class_names = [str(c) for c in sorted(np.unique(y))]
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.2, random_state=seed, stratify=y)
    x_tr, x_va, y_tr, y_va = train_test_split(x_tr, y_tr, test_size=0.2, random_state=seed, stratify=y_tr)
    scaler = MinMaxScaler()
    x_tr = scaler.fit_transform(x_tr).astype(np.float32)
    x_va = scaler.transform(x_va).astype(np.float32)
    x_te = scaler.transform(x_te).astype(np.float32)
    names = [f"f{i}" for i in range(x.shape[1])]
    return _to_unified("local/banknote", "local", "banknote", x_tr, x_va, x_te, y_tr, y_va, y_te, names, class_names)


def load_wine_splits(*, seed: int = 42) -> UnifiedSplits:
    path = DATASET_ROOT / "wine" / "wine.data"
    cols = [
        "class",
        "alcohol",
        "malic_acid",
        "ash",
        "alcalinity",
        "magnesium",
        "phenols",
        "flavanoids",
        "nonflavanoid_phenols",
        "proanthocyanins",
        "color_intensity",
        "hue",
        "od280_od315",
        "proline",
    ]
    df = pd.read_csv(path, header=None, names=cols)
    y = (df["class"].values - 1).astype(np.int64)
    class_names = [str(c) for c in sorted(df["class"].unique())]
    x_df = df.drop(columns=["class"])
    x_tr, x_te, y_tr, y_te = train_test_split(
        x_df.values, y, test_size=0.2, random_state=seed, stratify=y
    )
    x_tr, x_va, y_tr, y_va = train_test_split(
        x_tr, y_tr, test_size=0.2, random_state=seed, stratify=y_tr
    )
    scaler = MinMaxScaler()
    x_tr = scaler.fit_transform(x_tr).astype(np.float32)
    x_va = scaler.transform(x_va).astype(np.float32)
    x_te = scaler.transform(x_te).astype(np.float32)
    return _to_unified(
        "local/wine", "local", "wine", x_tr, x_va, x_te, y_tr, y_va, y_te, list(x_df.columns), class_names
    )


def load_titanic_splits(*, seed: int = 42) -> UnifiedSplits:
    path = DATASET_ROOT / "titanic" / "train.csv"
    df = pd.read_csv(path)
    df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"], errors="ignore")
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode().iloc[0])
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())
    y = df["Survived"].values.astype(np.int64)
    x_df = pd.get_dummies(df.drop(columns=["Survived"]), drop_first=True)
    class_names = ["0", "1"]
    x_tr, x_te, y_tr, y_te = train_test_split(
        x_df.values, y, test_size=0.2, random_state=seed, stratify=y
    )
    x_tr, x_va, y_tr, y_va = train_test_split(
        x_tr, y_tr, test_size=0.2, random_state=seed, stratify=y_tr
    )
    scaler = MinMaxScaler()
    x_tr = scaler.fit_transform(x_tr).astype(np.float32)
    x_va = scaler.transform(x_va).astype(np.float32)
    x_te = scaler.transform(x_te).astype(np.float32)
    return _to_unified(
        "local/titanic", "local", "titanic", x_tr, x_va, x_te, y_tr, y_va, y_te, list(x_df.columns), class_names
    )


def load_heart_splits(*, seed: int = 42) -> UnifiedSplits:
    path = DATASET_ROOT / "heart+disease" / "processed.cleveland.data"
    cols = [
        "age",
        "sex",
        "cp",
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalach",
        "exang",
        "oldpeak",
        "slope",
        "ca",
        "thal",
        "target",
    ]
    df = pd.read_csv(path, header=None, names=cols)
    df = df.replace("?", np.nan)
    for c in df.columns:
        if c != "target":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna().reset_index(drop=True)
    y_raw = (df["target"] > 0).astype(np.int64).values
    x_df = df.drop(columns=["target"])
    class_names = ["0", "1"]
    x_tr, x_te, y_tr, y_te = train_test_split(
        x_df.values, y_raw, test_size=0.2, random_state=seed, stratify=y_raw
    )
    x_tr, x_va, y_tr, y_va = train_test_split(
        x_tr, y_tr, test_size=0.2, random_state=seed, stratify=y_tr
    )
    scaler = MinMaxScaler()
    x_tr = scaler.fit_transform(x_tr).astype(np.float32)
    x_va = scaler.transform(x_va).astype(np.float32)
    x_te = scaler.transform(x_te).astype(np.float32)
    return _to_unified(
        "local/heart", "local", "heart", x_tr, x_va, x_te, y_tr, y_va, y_te, list(x_df.columns), class_names
    )


def load_diabetes_splits(*, seed: int = 42) -> UnifiedSplits:
    path = DATASET_ROOT / "archive (3)" / "diabetes.csv"
    df = pd.read_csv(path)
    y_raw = df["Outcome"].values
    x_df = df.drop(columns=["Outcome"])
    le = LabelEncoder()
    y = le.fit_transform(y_raw.astype(str))
    class_names = [str(c) for c in le.classes_]
    x_tr, x_te, y_tr, y_te = train_test_split(
        x_df.values, y, test_size=0.2, random_state=seed, stratify=y
    )
    x_tr, x_va, y_tr, y_va = train_test_split(
        x_tr, y_tr, test_size=0.2, random_state=seed, stratify=y_tr
    )
    scaler = MinMaxScaler()
    x_tr = scaler.fit_transform(x_tr).astype(np.float32)
    x_va = scaler.transform(x_va).astype(np.float32)
    x_te = scaler.transform(x_te).astype(np.float32)
    names = list(x_df.columns)
    return _to_unified(
        "local/diabetes", "local", "diabetes", x_tr, x_va, x_te, y_tr, y_va, y_te, names, class_names
    )


_LOCAL_LOADERS = {
    "local/dry_bean": load_dry_bean_splits,
    "local/wdbc": load_wdbc_splits,
    "local/banknote": load_banknote_splits,
    "local/diabetes": load_diabetes_splits,
    "local/wine": load_wine_splits,
    "local/titanic": load_titanic_splits,
    "local/heart": load_heart_splits,
}

_LOCAL_PATHS = {
    "local/dry_bean": DATASET_ROOT / "dry+bean+dataset" / "DryBeanDataset" / "Dry_Bean_Dataset.arff",
    "local/wdbc": DATASET_ROOT / "breast+cancer+wisconsin+diagnostic" / "wdbc.data",
    "local/banknote": DATASET_ROOT / "banknote+authentication" / "data_banknote_authentication.txt",
    "local/diabetes": DATASET_ROOT / "archive (3)" / "diabetes.csv",
    "local/wine": DATASET_ROOT / "wine" / "wine.data",
    "local/titanic": DATASET_ROOT / "titanic" / "train.csv",
    "local/heart": DATASET_ROOT / "heart+disease" / "processed.cleveland.data",
}


def discover_all_dataset_ids(
    *,
    sources: list[str] | None = None,
    skip_amazon: bool = False,
) -> list[str]:
    sources = sources or ["dlbac", "hyconex", "hyperlogic", "local"]
    ids: list[str] = []

    if "dlbac" in sources:
        for spec in discover_dlbac_datasets():
            if not spec.has_train:
                continue
            if skip_amazon and spec.name.startswith("amazon"):
                continue
            ids.append(f"dlbac/{spec.name}")

    if "hyconex" in sources or "hyperlogic" in sources:
        from hyconex_hyperlogic_datasets import list_available_datasets

        avail = list_available_datasets()
        if "hyconex" in sources:
            ids.extend(f"hyconex/{n}" for n in avail["hyconex"])
        if "hyperlogic" in sources:
            ids.extend(f"hyperlogic/{n}" for n in avail["hyperlogic"])

    if "local" in sources:
        for key in _LOCAL_LOADERS:
            p = _LOCAL_PATHS.get(key)
            if p is not None and p.is_file():
                ids.append(key)

    return sorted(ids)


def load_splits(dataset_id: str, *, seed: int = 42) -> UnifiedSplits:
    if dataset_id.startswith("dlbac/"):
        name = dataset_id.split("/", 1)[1]
        specs = {s.name: s for s in discover_dlbac_datasets() if s.has_train}
        if name not in specs:
            raise KeyError(dataset_id)
        return load_dlbac_splits(specs[name], seed=seed)

    if dataset_id.startswith(("hyconex/", "hyperlogic/")):
        return load_hyconex_hyperlogic_splits(dataset_id, seed=seed)

    if dataset_id in _LOCAL_LOADERS:
        return _LOCAL_LOADERS[dataset_id](seed=seed)

    raise KeyError(f"Dataset inconnu: {dataset_id}")
