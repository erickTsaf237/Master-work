"""
DR-Net pur (100 % règles) sur DLBAC — pipeline DLBACα + HyperLogic.

Phase 1 : DR-Net seul (cf_lambda=0)
Phase 2 : greffe CF (HyperLogic binaire si dim<=512, HyConEx si Amazon 14k)

Usage :
    python train_pure_drnet_dlbac.py --dataset u4k-r4k-auth11k --save
    python train_pure_drnet_dlbac.py --dataset amazon1 --save
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from hyperlogic_pure import PureDRConfig, PureDRNetTrainer
from prepare_dlbac_datasets import (
    discover_dlbac_datasets,
    explain_counterfactual_continuous,
    format_rule,
    pick_counterfactual_example,
)
from train_nouveau_module_dlbac_quantile import build_onehot_splits

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results" / "pure_drnet_dlbac"

ACCEPT_AUROC_AMAZON = 0.65
ACCEPT_AUROC_SYNTH = 0.85
MIN_AUROC_FOR_CF = 0.50


def discover_specs():
    return [s for s in discover_dlbac_datasets() if s.has_train]


def fit_svm_teacher_proba(x_train: np.ndarray, y_train: np.ndarray, *, seed: int = 42) -> np.ndarray:
    """Probabilités SVM calibré sur le train (enseignant pour distillation)."""
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.svm import LinearSVC

    clf = CalibratedClassifierCV(
        LinearSVC(class_weight="balanced", max_iter=3000, random_state=seed),
        cv=3,
        method="sigmoid",
    )
    clf.fit(np.asarray(x_train, dtype=np.float32), y_train)
    return clf.predict_proba(x_train).astype(np.float32)


def config_for_dataset(
    name: str, n_features: int, num_classes: int, *, distill: bool = False
) -> PureDRConfig:
    is_amazon = name.startswith("amazon")
    high_dim = n_features > 512
    return PureDRConfig(
        seed=42,
        epochs=50 if is_amazon else 40,
        batch_size=96 if high_dim else 64,
        lr=1e-3 if not high_dim else 1e-3,
        num_rules=64 if is_amazon else 64,
        temperature=1.0 if is_amazon else 0.6,
        embed_dim_high=256,
        cf_lambda=0.0,
        flip_lambda=0.0,
        cf_epochs=12 if is_amazon else 10,
        cf_lambda_phase2=0.08,
        flip_lambda_phase2=0.04,
        early_stop_metric="auroc",
        max_instance_dim=512,
        distill_lambda=1.5 if (is_amazon and distill) else 0.0,
        distill_only=is_amazon and distill,
        use_class_weights=not (is_amazon and distill),
    )


def train_one(spec, *, save_dir: Path | None = None, verbose: bool = True, distill: bool = False) -> dict:
    splits = build_onehot_splits(spec, val_size=0.2, random_state=42, use_cache=True)

    if verbose:
        print(f"\n=== {spec.name} (DR-Net PUR) ===", flush=True)
        print(f"  features: {splits.x_train.shape[1]}", flush=True)
        print(f"  classes: {splits.num_classes}", flush=True)

    cfg = config_for_dataset(spec.name, splits.x_train.shape[1], splits.num_classes, distill=distill)
    trainer = PureDRNetTrainer(cfg)

    teacher_proba = None
    if distill and cfg.distill_lambda > 0:
        if verbose:
            print("  [Distillation] SVM calibré -> proba train...", flush=True)
        teacher_proba = fit_svm_teacher_proba(splits.x_train, splits.y_train, seed=cfg.seed)

    if verbose:
        print(f"  Device: {trainer.device}", flush=True)
        print(
            f"  Phase 1: DR-Net pur (100% règles)"
            + (f", distill_lambda={cfg.distill_lambda}" if teacher_proba is not None else ""),
            flush=True,
        )

    result = trainer.fit(
        splits.x_train,
        splits.y_train,
        x_val=splits.x_val,
        y_val=splits.y_val,
        feature_names=splits.feature_names,
        class_names=splits.class_names,
        verbose=verbose,
        phase="drnet",
        teacher_proba=teacher_proba,
    )

    metrics_p1 = trainer.evaluate(splits.x_test, splits.y_test, counterfactuals=False)
    auroc_p1 = float(metrics_p1.get("auroc_ovr", 0.0))

    if verbose:
        mode = trainer.model.mode if trainer.model else "?"
        print(f"  Mode: {mode} | Phase 1 test AUROC: {auroc_p1:.4f}", flush=True)

    if auroc_p1 >= MIN_AUROC_FOR_CF and cfg.cf_epochs > 0:
        if verbose:
            print(f"  Phase 2: greffe CF ({cfg.cf_epochs} epochs)", flush=True)
        trainer.config = replace(
            cfg,
            cf_lambda=cfg.cf_lambda_phase2,
            flip_lambda=cfg.flip_lambda_phase2,
        )
        trainer.fit(
            splits.x_train,
            splits.y_train,
            x_val=splits.x_val,
            y_val=splits.y_val,
            feature_names=splits.feature_names,
            class_names=splits.class_names,
            verbose=verbose,
            resume=True,
            phase="cf",
        )

    metrics = trainer.evaluate(splits.x_test, splits.y_test, counterfactuals=True)
    rules = trainer.export_rules(top_per_rule=4, min_abs_weight=0.001)

    is_amazon = spec.name.startswith("amazon")
    threshold = ACCEPT_AUROC_AMAZON if is_amazon else ACCEPT_AUROC_SYNTH
    test_auroc = float(metrics.get("auroc_ovr", 0.0))

    example_rule = rules[0] if rules else None
    example_cf = None
    picked = pick_counterfactual_example(trainer, splits.x_test, splits.y_test)
    if picked is not None:
        idx, target = picked
        example_cf = explain_counterfactual_continuous(
            trainer, splits.x_test, idx, target, y_true=int(splits.y_test[idx])
        )

    model_path = None
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        model_path = save_dir / f"{spec.name}_model.pt"
        trainer.save_checkpoint(model_path)

    return {
        "dataset": spec.name,
        "model_mode": trainer.model.mode if trainer.model else None,
        "pure_drnet": True,
        "linear_head": False,
        "num_features": int(splits.x_train.shape[1]),
        "num_classes": splits.num_classes,
        "best_val_auroc": float(result.best_val_auroc),
        "phase1_test_auroc": auroc_p1,
        "test_accuracy": float(metrics["accuracy"]),
        "test_auroc": test_auroc,
        "acceptable": test_auroc >= threshold,
        "accept_threshold": threshold,
        "cf_validity": float(metrics.get("counterfactuals", {}).get("validity_cf", 0.0)),
        "n_rules": len(rules),
        "rules_top3": rules[:3],
        "example_rule": example_rule,
        "example_rule_text": format_rule(example_rule) if example_rule else None,
        "example_counterfactual": example_cf,
        "model_checkpoint": str(model_path) if model_path else None,
        "cf_type": "hyperlogic_binary"
        if trainer.model and trainer.model.mode in ("instance", "embed")
        else "hyconex_graft",
        "distillation": distill and cfg.distill_lambda > 0,
        "distill_lambda": cfg.distill_lambda,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="DR-Net pur sur DLBAC (DLBACα + HyperLogic)")
    p.add_argument("--dataset", nargs="*", default=["u4k-r4k-auth11k"])
    p.add_argument("--save", action="store_true")
    p.add_argument(
        "--distill",
        action="store_true",
        help="Distillation SVM -> DR-Net (recommandé pour Amazon)",
    )
    args = p.parse_args()

    specs = {s.name: s for s in discover_specs()}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for name in args.dataset:
        if name not in specs:
            raise SystemExit(f"Jeu introuvable: {name}")
        row = train_one(
            specs[name],
            save_dir=RESULTS_DIR if args.save else None,
            distill=args.distill,
        )
        if args.save:
            out = RESULTS_DIR / f"{name}_results.json"
            out.write_text(json.dumps(row, indent=2, default=str), encoding="utf-8")
        flag = "PASS" if row["acceptable"] else "FAIL"
        print(
            f"\n  [{flag}] {name} mode={row['model_mode']} auroc={row['test_auroc']:.4f} "
            f"rules={row['n_rules']} cf={row['cf_type']}",
            flush=True,
        )
        if row.get("example_rule_text"):
            print(f"  Règle: {row['example_rule_text']}", flush=True)


if __name__ == "__main__":
    main()
