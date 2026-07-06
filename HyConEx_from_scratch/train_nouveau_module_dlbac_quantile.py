"""
Pipeline DLBAC pour le nouveau_module :

1. Chargement des fichiers .sample bruts
2. Suppression des 2 premières colonnes (uid, rid) et des labels (opérations en fin de ligne)
3. Masquage des métadonnées cachées (comme DLBACα)
4. Normalisation MinMax sur les métadonnées (fit sur train)
5. Binarisation par quantiles (TabularBinarizer, mode quantize → littéraux bipolar {-1,+1})
6. Entraînement HybridDRNetModel

Pas de OneHotEncoder sklearn : l'encodage « type one-hot » est fait par nos bins/quantiles.

Usage :
    python train_nouveau_module_dlbac_quantile.py --dataset u4k-r4k-auth11k
    python train_nouveau_module_dlbac_quantile.py --all
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from nouveau_module import HybridDRConfig, HybridDRTrainer
from prepare_dlbac_datasets import (
    DLBAC_ROOT,
    DLBACDatasetSpec,
    apply_metadata_mask,
    binary_access_label,
    class_names_for_labels,
    discover_dlbac_datasets,
    joint_op_label,
    strip_uid_rid,
)

ROOT = Path(__file__).resolve().parent


@dataclass
class DlbacQuantileSplits:
    name: str
    kind: str
    label_mode: str
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    class_names: list[str]
    num_classes: int
    scaler: MinMaxScaler | None = None
    input_encoding: str = "quantile"
    onehot_dim_full: int | None = None


def load_dlbac_raw_arrays(
    train_path: Path,
    test_path: Path,
    *,
    num_ops: int = 4,
    label_mode: str = "joint_ops",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Charge train/test : X = métadonnées entières, y = labels (sans uid/rid ni colonnes ops)."""
    train_raw = np.loadtxt(train_path, dtype=np.float32)
    test_raw = np.loadtxt(test_path, dtype=np.float32)

    train_body = strip_uid_rid(train_raw)
    test_body = strip_uid_rid(test_raw)

    x_tr, ops_tr = apply_metadata_mask(train_body, num_ops=num_ops)
    x_te, ops_te = apply_metadata_mask(test_body, num_ops=num_ops)

    if label_mode == "joint_ops":
        y_tr = joint_op_label(ops_tr)
        y_te = joint_op_label(ops_te)
    else:
        y_tr = binary_access_label(ops_tr)
        y_te = binary_access_label(ops_te)

    return (
        x_tr.astype(np.float32),
        y_tr.astype(np.int64),
        x_te.astype(np.float32),
        y_te.astype(np.int64),
    )


def build_quantile_splits(
    spec: DLBACDatasetSpec,
    *,
    val_size: float = 0.2,
    random_state: int = 42,
) -> DlbacQuantileSplits:
    if not spec.has_train:
        raise ValueError(f"Pas de fichier train pour {spec.name}")

    x_tr, y_tr, x_te, y_te = load_dlbac_raw_arrays(
        spec.train_path,  # type: ignore[arg-type]
        spec.test_path,
        num_ops=spec.num_ops,
        label_mode=spec.label_mode,
    )

    x_train, x_val, y_train, y_val = train_test_split(
        x_tr,
        y_tr,
        test_size=val_size,
        random_state=random_state,
        stratify=y_tr,
    )

    scaler = MinMaxScaler()
    x_train_n = scaler.fit_transform(x_train).astype(np.float32)
    x_val_n = scaler.transform(x_val).astype(np.float32)
    x_test_n = scaler.transform(x_te).astype(np.float32)

    n_meta = x_train_n.shape[1]
    feature_names = [f"meta_{i}" for i in range(n_meta)]
    num_classes = int(max(y_train.max(), y_te.max()) + 1)
    class_names = class_names_for_labels(
        np.concatenate([y_train, y_te]),
        spec.label_mode,  # type: ignore[arg-type]
    )

    return DlbacQuantileSplits(
        name=spec.name,
        kind=spec.kind,
        label_mode=spec.label_mode,
        x_train=x_train_n,
        y_train=y_train,
        x_val=x_val_n,
        y_val=y_val,
        x_test=x_test_n,
        y_test=y_te,
        feature_names=feature_names,
        class_names=class_names,
        num_classes=num_classes,
        scaler=scaler,
        input_encoding="quantile",
    )


def select_top_variance_features(
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    max_features: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Garde les colonnes one-hot les plus variables (comme le notebook DLBAC)."""
    var = np.var(x_train, axis=0)
    idx = np.argsort(var)[::-1][:max_features]
    return x_train[:, idx], x_val[:, idx], x_test[:, idx], idx


ONEHOT_CACHE_DIR = ROOT / "data" / "dlbac_prepared" / "onehot_cache"


def _onehot_cache_path(
    spec: DLBACDatasetSpec,
    *,
    max_features: int | None,
    val_size: float,
    random_state: int,
) -> Path:
    tag = "full" if max_features is None else str(max_features)
    return ONEHOT_CACHE_DIR / f"{spec.name}_onehot_{tag}_v{val_size}_s{random_state}.npz"


def build_onehot_splits(
    spec: DLBACDatasetSpec,
    *,
    val_size: float = 0.2,
    random_state: int = 42,
    max_features: int | None = None,
    use_cache: bool = True,
) -> DlbacQuantileSplits:
    """
    Encodage one-hot sklearn (comme DlbacAlpha), reduction optionnelle par variance.
    max_features=None : toutes les colonnes (recommande, aligne les baselines SVM).
    """
    from prepare_dlbac_datasets import prepare_dataset

    if not spec.has_train:
        raise ValueError(f"Pas de fichier train pour {spec.name}")

    cache_path = _onehot_cache_path(
        spec, max_features=max_features, val_size=val_size, random_state=random_state
    )
    if use_cache and cache_path.is_file():
        z = np.load(cache_path, allow_pickle=True)
        n_feat = int(z["x_train"].shape[1])
        return DlbacQuantileSplits(
            name=spec.name,
            kind=spec.kind,
            label_mode=spec.label_mode,
            x_train=z["x_train"],
            y_train=z["y_train"],
            x_val=z["x_val"],
            y_val=z["y_val"],
            x_test=z["x_test"],
            y_test=z["y_test"],
            feature_names=[f"oh_{i}" for i in range(n_feat)],
            class_names=list(z["class_names"]),
            num_classes=int(z["num_classes"]),
            scaler=None,
            input_encoding="onehot",
            onehot_dim_full=int(z["onehot_dim_full"]),
        )

    prep = prepare_dataset(spec, val_size=val_size, random_state=random_state)
    onehot_full = int(prep.x_train.shape[1])
    x_train, x_val, x_test = prep.x_train, prep.x_val, prep.x_test

    if max_features is not None and x_train.shape[1] > max_features:
        x_train, x_val, x_test, _ = select_top_variance_features(
            x_train, x_val, x_test, max_features
        )

    n_feat = x_train.shape[1]
    feature_names = [f"oh_{i}" for i in range(n_feat)]
    splits = DlbacQuantileSplits(
        name=spec.name,
        kind=spec.kind,
        label_mode=spec.label_mode,
        x_train=x_train,
        y_train=prep.y_train,
        x_val=x_val,
        y_val=prep.y_val,
        x_test=x_test,
        y_test=prep.y_test,
        feature_names=feature_names,
        class_names=prep.class_names,
        num_classes=prep.num_classes,
        scaler=None,
        input_encoding="onehot",
        onehot_dim_full=onehot_full,
    )

    if use_cache:
        ONEHOT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            x_train=splits.x_train,
            y_train=splits.y_train,
            x_val=splits.x_val,
            y_val=splits.y_val,
            x_test=splits.x_test,
            y_test=splits.y_test,
            class_names=np.array(splits.class_names, dtype=object),
            num_classes=splits.num_classes,
            onehot_dim_full=onehot_full,
        )

    return splits


def config_for_dlbac(name: str, num_classes: int, n_features: int) -> HybridDRConfig:
    is_amazon = name.startswith("amazon")
    epochs = 40 if is_amazon else 50
    batch = 256 if n_features < 64 else 128
    num_rules = min(128, 32 + 6 * num_classes) if not is_amazon else 48
    hidden = 128 if n_features < 128 else 96
    return HybridDRConfig(
        seed=42,
        epochs=epochs,
        batch_size=batch,
        lr=1e-3,
        num_rules=num_rules,
        hyper_hidden_dim=hidden,
        cf_hidden_dim=hidden,
        tabresnet_n_blocks=4,
        bins_per_feature=4,
        input_encoding="quantize",
        use_class_weights=True,
        early_stop_metric="auto",
        temperature=0.7,
        cf_lambda=0.06 if num_classes > 2 else 0.10,
        flip_lambda=0.03,
        rule_sparsity_lambda=0.001,
    )


def train_on_splits(
    splits: DlbacQuantileSplits,
    *,
    epochs: int | None = None,
    verbose: bool = True,
) -> dict:
    cfg = config_for_dlbac(splits.name, splits.num_classes, splits.x_train.shape[1])
    if epochs is not None:
        cfg.epochs = epochs
    trainer = HybridDRTrainer(cfg)

    result = trainer.fit(
        splits.x_train,
        splits.y_train,
        x_val_cont=splits.x_val,
        y_val=splits.y_val,
        feature_names=splits.feature_names,
        class_names=splits.class_names,
        verbose=verbose,
    )

    metrics = trainer.evaluate(splits.x_test, splits.y_test, counterfactuals=True)
    rules = trainer.export_rules(top_per_rule=4, min_abs_weight=0.05)

    summary = {
        "dataset": splits.name,
        "raw_metadata_dim": len(splits.feature_names),
        "binary_dim": trainer.model.input_dim_bin if trainer.model else None,
        "binarizer_mode": trainer.binarizer.mode_,
        "bins_per_feature": cfg.bins_per_feature,
        "best_val_accuracy": result.best_val_accuracy,
        "best_val_auroc": result.best_val_auroc,
        "test_accuracy": metrics["accuracy"],
        "test_auroc": metrics.get("auroc_ovr"),
        "cf_validity": metrics["counterfactuals"]["validity_cf"],
        "cf_changed_bits": metrics["counterfactuals"]["changed_bits_mean"],
        "n_rules": len(rules),
    }

    if verbose:
        print(f"\n=== {splits.name} (quantile -> bipolar) ===", flush=True)
        print(f"  Métadonnées brutes : {splits.x_train.shape[1]} cols")
        print(f"  Après binarizer    : {summary['binary_dim']} dims ({summary['binarizer_mode']})")
        print(f"  Val acc / AUROC    : {summary['best_val_accuracy']:.4f} / {summary['best_val_auroc']:.4f}")
        print(f"  Test acc / AUROC   : {summary['test_accuracy']:.4f} / {summary['test_auroc']}")
        print(f"  CF validité        : {summary['cf_validity']:.4f}")
        print(f"  Règles extraites   : {summary['n_rules']}")
        if rules:
            print(f"  Ex. règle : IF {' AND '.join(rules[0]['if'][:3])} ... THEN {rules[0]['then_class']}")

    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DLBAC -> normalisation -> bins quantiles -> nouveau_module")
    p.add_argument("--dataset", nargs="*", help="Nom(s) de jeu (ex: u4k-r4k-auth11k)")
    p.add_argument("--all", action="store_true", help="Tous les jeux avec train")
    p.add_argument("--dlbac-root", type=Path, default=DLBAC_ROOT)
    p.add_argument("--val-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=None, help="Surcharge le nombre d'époques")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    specs = discover_dlbac_datasets(args.dlbac_root)
    by_name = {s.name: s for s in specs if s.has_train}

    if args.all:
        names = sorted(by_name)
    elif args.dataset:
        missing = [n for n in args.dataset if n not in by_name]
        if missing:
            raise SystemExit(f"Jeux introuvables ou sans train: {missing}")
        names = list(args.dataset)
    else:
        names = ["u4k-r4k-auth11k"]

    results: list[dict] = []
    for name in names:
        print("\n" + "=" * 72, flush=True)
        print(f"Pipeline quantile : {name}", flush=True)
        print("=" * 72, flush=True)
        splits = build_quantile_splits(by_name[name], val_size=args.val_size, random_state=args.seed)
        results.append(train_on_splits(splits, epochs=args.epochs))

    print("\n" + "=" * 72)
    print("Récapitulatif")
    for r in results:
        print(
            f"  {r['dataset']:22s} acc={r['test_accuracy']:.4f} auroc={r['test_auroc']} "
            f"D_raw={r['raw_metadata_dim']} D_bin={r['binary_dim']}"
        )


if __name__ == "__main__":
    main()
