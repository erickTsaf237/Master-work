"""Chargeurs pour HyConEx/data et HyperLogic/data (preprocessing mixte continu + catégoriel)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from tabular_mixed_preprocessing import (
    MixedTabularPreprocessor,
    TabularDatasetSplits,
    _wrap_array_splits,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HYCONEX_ROOT = PROJECT_ROOT / "HyConEx"
HYCONEX_DATA = HYCONEX_ROOT / "data"
HYPERLOGIC_DATA = PROJECT_ROOT / "HyperLogic" / "data"


def _import_hyconex_class(module: str, class_name: str):
    if str(HYCONEX_ROOT) not in sys.path:
        sys.path.insert(0, str(HYCONEX_ROOT))
    path = HYCONEX_ROOT / "counterfactuals" / "datasets" / f"{module}.py"
    spec = importlib.util.spec_from_file_location(f"hyconex_ds_{module}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Impossible de charger {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, class_name)


def _encode_labels(y: np.ndarray) -> tuple[np.ndarray, list[str]]:
    y = np.asarray(y)
    if np.issubdtype(y.dtype, np.integer) and y.min() >= 0:
        classes = sorted(np.unique(y))
        return y.astype(np.int64), [str(c) for c in classes]
    le = LabelEncoder()
    y_enc = le.fit_transform(y.astype(str))
    return y_enc.astype(np.int64), [str(c) for c in le.classes_]


def _split_xy(
    x: np.ndarray | pd.DataFrame,
    y: np.ndarray,
    *,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray | pd.DataFrame, ...]:
    x_train_full, x_test, y_train_full, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state, stratify=y
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full,
        y_train_full,
        test_size=val_size,
        random_state=random_state,
        stratify=y_train_full,
    )
    return x_train, x_val, x_test, y_train, y_val, y_test


def _from_mixed_dataframe(
    name: str,
    *,
    x_train_df: pd.DataFrame,
    x_val_df: pd.DataFrame,
    x_test_df: pd.DataFrame,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    continuous_cols: list[str],
    categorical_cols: list[str],
    class_names: list[str],
    bins_per_feature: int = 4,
) -> TabularDatasetSplits:
    prep = MixedTabularPreprocessor(bins_per_feature=bins_per_feature)
    prep.fit(x_train_df, continuous_cols=continuous_cols, categorical_cols=categorical_cols)
    return _wrap_array_splits(
        name,
        x_train=prep.transform(x_train_df),
        x_val=prep.transform(x_val_df),
        x_test=prep.transform(x_test_df),
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        feature_names=prep.feature_names_,
        class_names=class_names,
        continuous_cols=continuous_cols,
        categorical_cols=categorical_cols,
        input_encoding="bipolar",
    )


def _from_continuous_only(
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
) -> TabularDatasetSplits:
    scaler = MinMaxScaler()
    x_train_s = scaler.fit_transform(x_train).astype(np.float32)
    x_val_s = scaler.transform(x_val).astype(np.float32)
    x_test_s = scaler.transform(x_test).astype(np.float32)
    return _wrap_array_splits(
        name,
        x_train=x_train_s,
        x_val=x_val_s,
        x_test=x_test_s,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        feature_names=feature_names,
        class_names=class_names,
        continuous_cols=feature_names,
        categorical_cols=[],
        input_encoding="quantize",
    )


def _from_binary_matrix(
    name: str,
    *,
    x: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    class_names: list[str],
    random_state: int = 42,
) -> TabularDatasetSplits:
    y, class_names = _encode_labels(y)
    x_train, x_val, x_test, y_train, y_val, y_test = _split_xy(
        x.astype(np.float32), y, random_state=random_state
    )
    return _wrap_array_splits(
        name,
        x_train=x_train,
        x_val=x_val,
        x_test=x_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        feature_names=feature_names,
        class_names=class_names,
        continuous_cols=[],
        categorical_cols=feature_names,
        input_encoding="bipolar",
    )


def _load_hyconex_counterfactuals_dataset(
    name: str,
    *,
    module: str,
    class_name: str,
    filename: str,
    random_state: int = 42,
) -> TabularDatasetSplits:
    csv_path = HYCONEX_DATA / filename
    if not csv_path.is_file():
        raise FileNotFoundError(f"Fichier manquant: {csv_path}")

    ds_cls = _import_hyconex_class(module, class_name)
    base_cls = _import_hyconex_class("base", "AbstractDataset")
    ds = ds_cls.__new__(ds_cls)
    base_cls.__init__(ds, data=None)
    raw = ds.load(str(csv_path))
    x, y = ds.preprocess(raw)

    if hasattr(ds, "feature_columns"):
        feature_columns = list(ds.feature_columns)
    else:
        feature_columns = [f"f{i}" for i in range(x.shape[1])]

    num_idx = list(getattr(ds, "numerical_columns", []) or [])
    cat_idx = list(getattr(ds, "categorical_columns", []) or [])
    cont_cols = [feature_columns[i] for i in num_idx]
    cat_cols = [feature_columns[i] for i in cat_idx]

    df = pd.DataFrame(x, columns=feature_columns)
    y, class_names = _encode_labels(y)
    x_train_df, x_val_df, x_test_df, y_train, y_val, y_test = _split_xy(
        df, y, random_state=random_state
    )

    if cat_cols:
        return _from_mixed_dataframe(
            name,
            x_train_df=x_train_df,
            x_val_df=x_val_df,
            x_test_df=x_test_df,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
            continuous_cols=cont_cols,
            categorical_cols=cat_cols,
            class_names=class_names,
        )

    x_train = x_train_df[cont_cols].astype(np.float32).to_numpy()
    x_val = x_val_df[cont_cols].astype(np.float32).to_numpy()
    x_test = x_test_df[cont_cols].astype(np.float32).to_numpy()
    return _from_continuous_only(
        name,
        x_train=x_train,
        x_val=x_val,
        x_test=x_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        feature_names=cont_cols,
        class_names=class_names,
    )


def load_moons_with_blob_tabular(random_state: int = 42) -> TabularDatasetSplits:
    csv_path = HYCONEX_DATA / "moons_with_blob.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Fichier manquant: {csv_path}")
    raw = pd.read_csv(csv_path)
    feature_columns = list(raw.columns[:-1])
    x = raw[feature_columns].astype(np.float32).to_numpy()
    y, class_names = _encode_labels(raw[raw.columns[-1]].to_numpy())
    x_train, x_val, x_test, y_train, y_val, y_test = _split_xy(x, y, random_state=random_state)
    return _from_continuous_only(
        "moons_with_blob",
        x_train=x_train,
        x_val=x_val,
        x_test=x_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        feature_names=feature_columns,
        class_names=class_names,
    )


def load_cardio_tabular(random_state: int = 42) -> TabularDatasetSplits:
    path = HYPERLOGIC_DATA / "cardio_train.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Fichier manquant: {path}")
    df = pd.read_csv(path, sep=";")
    df = df.drop(columns=["id"], errors="ignore")
    y, class_names = _encode_labels(df["cardio"].to_numpy())
    x_df = df.drop(columns=["cardio"])
    x_df = x_df.copy()
    x_df["age"] = x_df["age"] // 365

    continuous_cols = ["age", "height", "weight", "ap_hi", "ap_lo"]
    categorical_cols = ["gender", "cholesterol", "gluc", "smoke", "alco", "active"]

    x_train_df, x_val_df, x_test_df, y_train, y_val, y_test = _split_xy(
        x_df, y, random_state=random_state
    )
    return _from_mixed_dataframe(
        "cardio",
        x_train_df=x_train_df,
        x_val_df=x_val_df,
        x_test_df=x_test_df,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        continuous_cols=continuous_cols,
        categorical_cols=categorical_cols,
        class_names=class_names,
    )


def load_disease_tabular(random_state: int = 42) -> TabularDatasetSplits:
    path = HYPERLOGIC_DATA / "disease_symptom.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Fichier manquant: {path}")
    df = pd.read_csv(path)
    for col in df.columns:
        df[col] = df[col].astype(str).str.strip()
    df = df.replace({"": np.nan, "nan": np.nan})

    y, class_names = _encode_labels(df["Disease"].to_numpy())
    symptom_df = df.drop(columns=["Disease"]).fillna("<NA>")

    values: set[str] = set()
    for row in symptom_df.to_numpy().ravel():
        if pd.notna(row) and str(row) not in ("<NA>", "nan"):
            values.add(str(row))
    vocab = sorted(values)
    vocab_map = {v: i for i, v in enumerate(vocab)}

    x = np.zeros((len(symptom_df), len(vocab)), dtype=np.float32)
    for i, row in enumerate(symptom_df.to_numpy()):
        for val in row:
            if pd.notna(val) and str(val) in vocab_map:
                x[i, vocab_map[str(val)]] = 1.0

    feature_names = [f"symptom={v}" for v in vocab]
    return _from_binary_matrix(
        "disease",
        x=x,
        y=y,
        feature_names=feature_names,
        class_names=class_names,
        random_state=random_state,
    )


def load_brca_bin_tabular(*, max_genes: int = 120, random_state: int = 42) -> TabularDatasetSplits:
    data_path = HYPERLOGIC_DATA / "brca_processed_logtpm_balanced_75perc.tsv"
    flag_path = HYPERLOGIC_DATA / "brca_processed_logtpm_balanced_flag_tissue_origin.tsv"
    if not data_path.is_file() or not flag_path.is_file():
        raise FileNotFoundError(f"Fichiers BRCA-n manquants dans {HYPERLOGIC_DATA}")

    df = pd.read_csv(data_path, sep="\t")
    labels_raw = pd.read_csv(flag_path, sep="\t", header=None).iloc[:, 0].to_numpy()
    y = (labels_raw == "Tumor").astype(np.int64)
    class_names = ["normal", "tumor"]

    if max_genes > 0 and df.shape[1] > max_genes:
        var = df.var(axis=0).sort_values(ascending=False)
        keep = var.head(max_genes).index.tolist()
        df = df[keep]

    x = df.astype(np.float32).to_numpy()
    feature_names = list(df.columns)
    x_train, x_val, x_test, y_train, y_val, y_test = _split_xy(x, y, random_state=random_state)
    return _from_continuous_only(
        "brca-n",
        x_train=x_train,
        x_val=x_val,
        x_test=x_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        feature_names=feature_names,
        class_names=class_names,
    )


def load_brca_mult_tabular(*, max_genes: int = 120, random_state: int = 42) -> TabularDatasetSplits:
    data_path = HYPERLOGIC_DATA / "brca_processed_logtpm_tumor_noduplicates_75_perc_subtypebalanced.tsv"
    flag_path = HYPERLOGIC_DATA / "brca_subtype_flag.tsv"
    if not data_path.is_file() or not flag_path.is_file():
        raise FileNotFoundError(f"Fichiers BRCA-s manquants dans {HYPERLOGIC_DATA}")

    df = pd.read_csv(data_path, sep="\t")
    y = pd.read_csv(flag_path, sep="\t", header=None).iloc[:, 0].to_numpy().astype(np.int64) - 1
    class_names = [str(c) for c in sorted(np.unique(y))]

    if max_genes > 0 and df.shape[1] > max_genes:
        var = df.var(axis=0).sort_values(ascending=False)
        keep = var.head(max_genes).index.tolist()
        df = df[keep]

    x = df.astype(np.float32).to_numpy()
    feature_names = list(df.columns)
    x_train, x_val, x_test, y_train, y_val, y_test = _split_xy(x, y, random_state=random_state)
    return _from_continuous_only(
        "brca-s",
        x_train=x_train,
        x_val=x_val,
        x_test=x_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        feature_names=feature_names,
        class_names=class_names,
    )


HYCONEX_DATASET_SPECS: dict[str, tuple[str, str, str]] = {
    "adult": ("adult", "AdultDataset", "adult.csv"),
    "audit": ("audit", "AuditDataset", "audit.csv"),
    "blobs": ("blobs", "BlobsDataset", "blobs.csv"),
    "compas": ("compas", "CompasDataset", "compas_two_years.csv"),
    "german_credit": ("german_credit", "GermanCreditDataset", "german_credit.csv"),
    "heloc": ("heloc", "HelocDataset", "heloc.csv"),
    "law": ("law", "LawDataset", "law.csv"),
    "moons": ("moons", "MoonsDataset", "moons.csv"),
    "wine": ("wine", "WineDataset", "wine.csv"),
}


def load_hyconex_tabular(name: str, random_state: int = 42) -> TabularDatasetSplits:
    if name == "moons_with_blob":
        return load_moons_with_blob_tabular(random_state=random_state)
    if name not in HYCONEX_DATASET_SPECS:
        raise KeyError(f"Jeu HyConEx inconnu: {name}")
    module, cls_name, filename = HYCONEX_DATASET_SPECS[name]
    return _load_hyconex_counterfactuals_dataset(
        name,
        module=module,
        class_name=cls_name,
        filename=filename,
        random_state=random_state,
    )


def load_hyperlogic_tabular(name: str, random_state: int = 42) -> TabularDatasetSplits:
    loaders = {
        "cardio": load_cardio_tabular,
        "disease": load_disease_tabular,
        "brca-n": load_brca_bin_tabular,
        "brca-s": load_brca_mult_tabular,
    }
    if name not in loaders:
        raise KeyError(f"Jeu HyperLogic inconnu: {name}")
    return loaders[name](random_state=random_state)


def list_available_datasets() -> dict[str, list[str]]:
    hyconex = sorted(HYCONEX_DATASET_SPECS.keys()) + ["moons_with_blob"]
    hyperlogic = ["cardio", "disease", "brca-n", "brca-s"]
    available_hyconex: list[str] = []
    available_hyperlogic: list[str] = []

    for name in hyconex:
        if name == "moons_with_blob":
            ok = (HYCONEX_DATA / "moons_with_blob.csv").is_file()
        else:
            ok = (HYCONEX_DATA / HYCONEX_DATASET_SPECS[name][2]).is_file()
        if ok:
            available_hyconex.append(name)

    hyper_files = {
        "cardio": "cardio_train.csv",
        "disease": "disease_symptom.csv",
        "brca-n": "brca_processed_logtpm_balanced_75perc.tsv",
        "brca-s": "brca_processed_logtpm_tumor_noduplicates_75_perc_subtypebalanced.tsv",
    }
    for name, fname in hyper_files.items():
        if (HYPERLOGIC_DATA / fname).is_file():
            available_hyperlogic.append(name)

    return {"hyconex": available_hyconex, "hyperlogic": available_hyperlogic}


def build_all_dataset_loaders() -> dict[str, Callable[[], TabularDatasetSplits]]:
    avail = list_available_datasets()
    loaders: dict[str, Callable[[], TabularDatasetSplits]] = {}
    for name in avail["hyconex"]:
        loaders[f"hyconex/{name}"] = (lambda n=name: load_hyconex_tabular(n))
    for name in avail["hyperlogic"]:
        loaders[f"hyperlogic/{name}"] = (lambda n=name: load_hyperlogic_tabular(n))
    return loaders
