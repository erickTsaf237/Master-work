"""
Entrainement HyConEx + HyperLogic sur DLBAC (synthetique + Amazon).

Objectif : AUROC >= 0.75 (Amazon), >= 0.90 (synthetique 16 classes).

Usage :
    python train_hyconex_hyperlogic_dlbac.py --dataset amazon1
    python train_hyconex_hyperlogic_dlbac.py --dataset u4k-r4k-auth11k
    python train_hyconex_hyperlogic_dlbac.py --all-synthetic
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from hyconex_hyperlogic import HybridConfig, HyConExHyperLogicTrainer
from nouveau_module.binary_metrics import predict_with_grant_threshold, tune_grant_threshold
from nouveau_module.sklearn_baseline import train_linear_svm_baseline
from prepare_dlbac_datasets import (
    discover_dlbac_datasets,
    explain_counterfactual_continuous,
    format_rule,
    pick_counterfactual_example,
)
from train_nouveau_module_dlbac_quantile import build_onehot_splits

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results" / "hyconex_hyperlogic_dlbac"

ACCEPT_AUROC_AMAZON = 0.72
ACCEPT_AUROC_SYNTH = 0.88


def discover_specs():
    return [s for s in discover_dlbac_datasets() if s.has_train]


def config_for_dataset(name: str, n_features: int, num_classes: int) -> HybridConfig:
    is_amazon = name.startswith("amazon")
    high_dim = n_features > 1000
    return HybridConfig(
        seed=42,
        epochs=35 if is_amazon else 30,
        batch_size=128 if high_dim else 128,
        lr=3e-3 if high_dim else 1e-3,
        embed_dim=128 if high_dim else 128,
        num_rules=48 if is_amazon else 64,
        temperature=0.5,
        linear_weight=0.7 if high_dim else 0.55,
        rule_weight=0.3 if high_dim else 0.45,
        cf_lambda=0.0,
        cf_epochs=8 if is_amazon else 10,
        cf_lambda_phase2=0.06,
        flip_lambda_phase2=0.02,
    )


def train_one(
    spec,
    *,
    max_features: int | None = None,
    use_cache: bool = True,
    run_baseline: bool = False,
    verbose: bool = True,
    save_dir: Path | None = None,
) -> dict:
    splits = build_onehot_splits(
        spec, val_size=0.2, random_state=42, max_features=max_features, use_cache=use_cache
    )

    if verbose:
        print(f"\n=== {spec.name} ===", flush=True)
        print(f"  features: {splits.x_train.shape[1]} (onehot_full={splits.onehot_dim_full})", flush=True)
        print(f"  classes: {splits.num_classes}", flush=True)

    baselines = {}
    if run_baseline and splits.num_classes == 2:
        if verbose:
            print("  [Baseline LinearSVM]...", flush=True)
        baselines["svm"] = train_linear_svm_baseline(
            splits.x_train,
            splits.y_train,
            splits.x_val,
            splits.y_val,
            splits.x_test,
            splits.y_test,
        )
        if verbose:
            print(f"    AUROC={baselines['svm']['test_auroc']:.4f}", flush=True)

    cfg = config_for_dataset(spec.name, splits.x_train.shape[1], splits.num_classes)
    trainer = HyConExHyperLogicTrainer(cfg)

    if verbose:
        print(f"  Device: {trainer.device}", flush=True)
        print(f"  Phase 1: classification (linear + rules)", flush=True)

    result = trainer.fit(
        splits.x_train,
        splits.y_train,
        x_val=splits.x_val,
        y_val=splits.y_val,
        feature_names=splits.feature_names,
        class_names=splits.class_names,
        verbose=verbose,
    )

    metrics_p1 = trainer.evaluate(splits.x_test, splits.y_test, counterfactuals=False)
    auroc_p1 = metrics_p1.get("auroc_ovr", 0.0)

    if verbose:
        print(f"  Phase 1 test AUROC: {auroc_p1:.4f}", flush=True)

    if auroc_p1 >= 0.65 and cfg.cf_epochs > 0:
        if verbose:
            print(f"  Phase 2: CF ({cfg.cf_epochs} epochs)", flush=True)
        trainer.config = replace(
            cfg,
            epochs=cfg.cf_epochs,
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
        )

    metrics = trainer.evaluate(splits.x_test, splits.y_test, counterfactuals=True)
    thresh, tune = None, None
    if splits.num_classes == 2:
        proba_val = trainer.predict_proba(splits.x_val)
        thresh, tune = tune_grant_threshold(proba_val, splits.y_val, metric="deny_f1")

    rules = trainer.export_rules(top_per_rule=4, min_abs_weight=0.001)
    is_amazon = spec.name.startswith("amazon")
    threshold = ACCEPT_AUROC_AMAZON if is_amazon else ACCEPT_AUROC_SYNTH
    test_auroc = float(metrics.get("auroc_ovr", 0.0))
    ok = test_auroc >= threshold

    example_rule = rules[0] if rules else None
    example_cf = None
    picked = pick_counterfactual_example(trainer, splits.x_test, splits.y_test)
    if picked is not None:
        idx, target = picked
        example_cf = explain_counterfactual_continuous(
            trainer,
            splits.x_test,
            idx,
            target,
            y_true=int(splits.y_test[idx]),
        )

    model_path = None
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        model_path = save_dir / f"{spec.name}_model.pt"
        trainer.save_checkpoint(model_path)

    summary = {
        "dataset": spec.name,
        "num_features": int(splits.x_train.shape[1]),
        "num_classes": splits.num_classes,
        "baselines": baselines,
        "best_val_auroc": float(result.best_val_auroc),
        "test_accuracy": float(metrics["accuracy"]),
        "test_auroc": test_auroc,
        "acceptable": ok,
        "accept_threshold": threshold,
        "grant_threshold": float(thresh) if thresh is not None else None,
        "cf_validity": float(metrics.get("counterfactuals", {}).get("validity_cf", 0.0)),
        "n_rules": len(rules),
        "rules_top5": rules[:5],
        "example_rule": example_rule,
        "example_rule_text": format_rule(example_rule) if example_rule else None,
        "example_counterfactual": example_cf,
        "model_checkpoint": str(model_path) if model_path is not None else None,
    }

    if verbose:
        status = "OK" if ok else "BELOW TARGET"
        print(f"\n  >>> {status} test_auroc={test_auroc:.4f} (seuil {threshold})", flush=True)
        if baselines.get("svm"):
            print(f"      ref SVM={baselines['svm']['test_auroc']:.4f}", flush=True)
        print(f"      test_acc={summary['test_accuracy']:.4f} cf_valid={summary['cf_validity']:.4f}", flush=True)

    return summary


def parse_args():
    p = argparse.ArgumentParser(description="HyConEx+HyperLogic sur DLBAC")
    p.add_argument("--dataset", nargs="*", help="Noms de jeux (ex: amazon1 u4k-r4k-auth11k)")
    p.add_argument("--all-synthetic", action="store_true")
    p.add_argument("--all-amazon", action="store_true")
    p.add_argument("--max-features", type=int, default=0, help="0=one-hot complet")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--baseline", action="store_true", help="LinearSVM (lent, ~1.5 Go RAM sur amazon1)")
    p.add_argument("--save", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    specs = {s.name: s for s in discover_specs()}
    max_feat = None if args.max_features <= 0 else args.max_features

    if args.all_synthetic:
        names = [n for n, s in specs.items() if s.kind == "synthetic"]
    elif args.all_amazon:
        names = [n for n, s in specs.items() if s.kind == "real_world"]
    elif args.dataset:
        names = list(args.dataset)
    else:
        names = ["amazon1", "u4k-r4k-auth11k"]

    missing = [n for n in names if n not in specs]
    if missing:
        raise SystemExit(f"Jeux introuvables: {missing}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in names:
        save_dir = RESULTS_DIR if args.save else None
        row = train_one(
            specs[name],
            max_features=max_feat,
            use_cache=not args.no_cache,
            run_baseline=args.baseline,
            save_dir=save_dir,
        )
        rows.append(row)
        if args.save:
            out = RESULTS_DIR / f"{name}_results.json"
            out.write_text(json.dumps(row, indent=2, default=str), encoding="utf-8")
            if row.get("example_rule_text"):
                print(f"  Exemple regle: {row['example_rule_text']}", flush=True)
            if row.get("example_counterfactual"):
                cf = row["example_counterfactual"]
                print(
                    f"  Exemple CF idx={cf['sample_idx']}: "
                    f"{cf['y_pred_orig_name']} -> {cf['y_pred_cf_name']} "
                    f"(valid={cf['valid']}, {cf['n_changes']} changements)",
                    flush=True,
                )

    print("\n" + "=" * 60, flush=True)
    print("RECAPITULATIF", flush=True)
    for r in rows:
        flag = "PASS" if r["acceptable"] else "FAIL"
        print(
            f"  [{flag}] {r['dataset']:22s} auroc={r['test_auroc']:.4f} "
            f"acc={r['test_accuracy']:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
