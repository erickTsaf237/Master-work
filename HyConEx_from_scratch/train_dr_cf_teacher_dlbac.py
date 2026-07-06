"""
DR-Net TabResNet (base Dry Bean) avec entrainement 2 phases :
  Phase 1 : la voie contrefactuelle predit
  Phase 2 : la tete regles apprend du teacher CF (distillation)

Usage :
    python train_dr_cf_teacher_dlbac.py --dataset u4k-r4k-auth11k amazon1 --save
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from dr_cf_teacher import CFTeacherDRConfig, CFTeacherDRTrainer
from prepare_dlbac_datasets import discover_dlbac_datasets, format_rule
from train_nouveau_module_dlbac_quantile import build_onehot_splits

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results" / "dr_cf_teacher_dlbac"


def discover_specs():
    return [s for s in discover_dlbac_datasets() if s.has_train]


def config_for_dataset(name: str, n_features: int, num_classes: int) -> CFTeacherDRConfig:
    is_amazon = name.startswith("amazon")
    high = n_features > 1000
    return CFTeacherDRConfig(
        seed=42,
        phase1_epochs=12 if not high else 8,
        phase2_epochs=10 if not high else 6,
        batch_size=64 if not high else 32,
        lr=1e-3 if not high else 8e-4,
        lr_phase2=5e-4 if not high else 4e-4,
        num_rules=64 if not high else 48,
        hyper_hidden_dim=128 if not high else 96,
        cf_hidden_dim=128 if not high else 96,
        temperature=0.7 if not high else 0.5,
        cf_lambda=0.35 if not high else 0.25,
        flip_lambda=0.04 if not high else 0.02,
        cf_predict_lambda=1.0,
        distill_lambda=1.5 if not high else 2.0,
        distill_temperature=2.0,
        embed_dim_high=128,
        max_instance_dim=512,
        input_encoding="bipolar",
    )


def train_one(spec, *, save_dir: Path | None = None, verbose: bool = True) -> dict:
    splits = build_onehot_splits(spec, val_size=0.2, random_state=42, use_cache=True)
    cfg = config_for_dataset(spec.name, splits.x_train.shape[1], splits.num_classes)
    trainer = CFTeacherDRTrainer(cfg)
    trainer.feature_names = splits.feature_names

    if verbose:
        mode = "instance/TabResNet" if splits.x_train.shape[1] <= cfg.max_instance_dim else "embed"
        print(f"\n=== {spec.name} (DR-CF-Teacher) ===", flush=True)
        print(
            f"  features: {splits.x_train.shape[1]} | classes: {splits.num_classes} | "
            f"mode: {mode} | K={cfg.num_rules}",
            flush=True,
        )
        print(f"  Device: {trainer.device}", flush=True)

    result = trainer.fit(
        splits.x_train,
        splits.y_train,
        splits.x_val,
        splits.y_val,
        feature_names=splits.feature_names,
        class_names=splits.class_names,
        verbose=verbose,
    )

    metrics = trainer.evaluate(splits.x_test, splits.y_test, counterfactuals=True)
    rules = trainer.export_rules(top_per_rule=4, min_abs_weight=0.001)
    rules_top5 = [
        {"text": format_rule(r), "score": r["score"], "then_class": r["then_class"]}
        for r in rules[:5]
    ]

    row = {
        "dataset": spec.name,
        "model": "CFTeacherDRModel",
        "phase1_epochs": cfg.phase1_epochs,
        "phase2_epochs": cfg.phase2_epochs,
        "distill_lambda": cfg.distill_lambda,
        "num_rules": cfg.num_rules,
        "num_features": int(splits.x_train.shape[1]),
        "num_classes": splits.num_classes,
        "best_val_accuracy": float(result.best_val_accuracy),
        "best_val_auroc": float(result.best_val_auroc),
        "test_accuracy": float(metrics["accuracy"]),
        "test_auroc": metrics.get("rules_only_auroc"),
        "rules_only_accuracy": float(metrics["rules_only_accuracy"]),
        "rules_only_auroc": metrics.get("rules_only_auroc"),
        "cf_predict_accuracy": float(metrics["cf_predict_accuracy"]),
        "cf_predict_auroc": metrics.get("cf_predict_auroc"),
        "cf_validity": float(metrics["counterfactuals"]["validity_cf"]),
        "cf_changed_bits_mean": float(metrics["counterfactuals"]["changed_bits_mean"]),
        "n_rules_exported": len(rules),
        "rules_top5": rules_top5,
        "example_rule_text": format_rule(rules[0]) if rules else None,
        "acceptable": False,
    }

    is_amazon = spec.name.startswith("amazon")
    threshold = 0.72 if is_amazon else 0.88
    if row["test_auroc"] is not None:
        row["acceptable"] = float(row["test_auroc"]) >= threshold
    row["accept_threshold"] = threshold

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        ckpt = save_dir / f"{spec.name}_model.pt"
        torch.save(
            {
                "state_dict": trainer.model.state_dict(),
                "config": asdict(cfg),
                "class_names": splits.class_names,
                "feature_names": splits.feature_names,
            },
            ckpt,
        )
        row["model_checkpoint"] = str(ckpt)

    return row


def main() -> None:
    p = argparse.ArgumentParser(description="DR-CF-Teacher sur DLBAC")
    p.add_argument("--dataset", nargs="*", default=["u4k-r4k-auth11k", "amazon1"])
    p.add_argument("--save", action="store_true")
    args = p.parse_args()

    specs = {s.name: s for s in discover_specs()}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in args.dataset:
        if name not in specs:
            raise SystemExit(f"Jeu introuvable: {name}")
        row = train_one(specs[name], save_dir=RESULTS_DIR if args.save else None)
        rows.append(row)
        if args.save:
            (RESULTS_DIR / f"{name}_results.json").write_text(
                json.dumps(row, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
        flag = "PASS" if row["acceptable"] else "FAIL"
        print(
            f"\n  [{flag}] {name} mix/rules auroc={row['test_auroc']:.4f} "
            f"acc={row['test_accuracy']:.4f} | "
            f"rules_auroc={row['rules_only_auroc']:.4f} | "
            f"cf_pred_auroc={row['cf_predict_auroc']:.4f} | "
            f"cf_valid={row['cf_validity']:.4f}",
            flush=True,
        )
        if row.get("example_rule_text"):
            print(f"  Regle: {row['example_rule_text']}", flush=True)

    if args.save:
        (RESULTS_DIR / "summary.json").write_text(
            json.dumps(rows, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
