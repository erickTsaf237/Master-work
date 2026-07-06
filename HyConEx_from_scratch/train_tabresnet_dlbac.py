"""
TabResNet + regles metier + contrefactuels sur DLBAC (strategie eprouvee).

- u4k / basse dim : TabResNet instance (nouveau_module / Dry Bean)
- Amazon / haute dim : DR-Net embed + distillation SVM + greffe CF

Usage :
    python train_tabresnet_dlbac.py --dataset u4k-r4k-auth11k amazon1 --save
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from prepare_dlbac_datasets import (
    discover_dlbac_datasets,
    explain_counterfactual_continuous,
    explain_counterfactual_flip,
    format_rule,
    pick_counterfactual_example,
)
from hyconex_pure_bipolar import explain_counterfactual_bipolar
from hyconex_pure_bipolar.bipolar import continuous_to_bipolar
from tabresnet_dlbac import TabResNetDLBACConfig, TabResNetDLBACTrainer
from train_nouveau_module_dlbac_quantile import build_onehot_splits

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results" / "tabresnet_dlbac"


def pick_cf_example(trainer, mode, x_test, y_test, class_names, max_probe: int = 32):
    th = torch  # alias local : evite UnboundLocalError si import cache obsolete
    if mode == "embed":
        return pick_counterfactual_example(trainer, x_test, y_test, max_probe=max_probe)
    if mode == "bipolar_hyper":
        x_bin = continuous_to_bipolar(np.asarray(x_test[:max_probe], dtype=np.float32))
        with th.no_grad():
            x_t = th.tensor(x_bin, dtype=th.float32, device=trainer.device)
            preds = trainer.model(x_t).argmax(dim=1).cpu().numpy()
        for i in range(min(max_probe, len(y_test))):
            for target in range(len(class_names)):
                if target == int(preds[i]):
                    continue
                cf = explain_counterfactual_bipolar(
                    trainer._bipolar, x_bin, i, target,
                    y_true=int(y_test[i]),
                    feature_names=trainer.feature_names,
                    class_names=class_names,
                )
                if cf.get("valid"):
                    return i, target
        return None

    n = min(max_probe, len(y_test))
    x_bin = trainer.binarizer.transform(np.asarray(x_test[:n], dtype=np.float32))
    with th.no_grad():
        x_t = th.tensor(x_bin, dtype=th.float32, device=trainer.device)
        preds = trainer.model.predict_logits(x_t).argmax(dim=1).cpu().numpy()
    for i in range(n):
        for target in range(len(class_names)):
            if target == int(preds[i]):
                continue
            cf = explain_counterfactual_flip(trainer, x_test, i, target, y_true=int(y_test[i]))
            if cf["valid"]:
                return i, target
    return None


def discover_specs():
    return [s for s in discover_dlbac_datasets() if s.has_train]


def config_for_dataset(name: str, n_features: int, num_classes: int) -> TabResNetDLBACConfig:
    is_amazon = name.startswith("amazon")
    high = n_features > 512
    return TabResNetDLBACConfig(
        seed=42,
        instance_epochs=40 if is_amazon else 25,
        instance_batch_size=128 if n_features < 64 else 128,
        instance_num_rules=min(128, 32 + 6 * num_classes) if not high else 48,
        instance_cf_lambda=0.06 if num_classes > 2 else 0.10,
        instance_flip_lambda=0.03,
        instance_cf_warmup=0,
        embed_epochs=40,
        embed_cf_epochs=12 if is_amazon else 10,
        embed_batch_size=96 if is_amazon else 64,
        embed_num_rules=48,
        embed_distill_lambda=1.5 if is_amazon else 0.0,
        hyper_hidden_dim=128 if n_features < 128 else 96,
        input_encoding="auto",
        early_stop_metric="auto",
    )


def collect_training_history(
    spec,
    *,
    save_dir: Path | None = None,
    verbose: bool = False,
) -> list[dict]:
    """Entraîne uniquement pour capturer l'historique epoch par epoch (courbes d'apprentissage)."""
    splits = build_onehot_splits(spec, val_size=0.2, random_state=42, use_cache=True)
    cfg = config_for_dataset(spec.name, splits.x_train.shape[1], splits.num_classes)
    trainer = TabResNetDLBACTrainer(cfg)
    if verbose:
        print(f"\n=== {spec.name} (historique) ===", flush=True)
    result = trainer.fit(
        splits.x_train,
        splits.y_train,
        splits.x_val,
        splits.y_val,
        feature_names=splits.feature_names,
        class_names=splits.class_names,
        verbose=verbose,
    )
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        hist_path = save_dir / f"{spec.name}_history.json"
        hist_path.write_text(json.dumps(result.history, indent=2), encoding="utf-8")
    return result.history


def train_one(spec, *, save_dir: Path | None = None, verbose: bool = True) -> dict:
    splits = build_onehot_splits(spec, val_size=0.2, random_state=42, use_cache=True)
    cfg = config_for_dataset(spec.name, splits.x_train.shape[1], splits.num_classes)
    trainer = TabResNetDLBACTrainer(cfg)

    if verbose:
        print(f"\n=== {spec.name} (TabResNet DLBAC) ===", flush=True)
        print(
            f"  features: {splits.x_train.shape[1]} | classes: {splits.num_classes} | "
            f"K={cfg.instance_num_rules if splits.x_train.shape[1] <= 512 else cfg.embed_num_rules}",
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

    example_cf = None
    picked = pick_cf_example(trainer, result.mode, splits.x_test, splits.y_test, splits.class_names)
    if picked is not None:
        idx, target = picked
        if result.mode == "instance":
            example_cf = explain_counterfactual_flip(
                trainer, splits.x_test, idx, target, y_true=int(splits.y_test[idx])
            )
        elif result.mode == "bipolar_hyper":
            x_bin = continuous_to_bipolar(splits.x_test)
            example_cf = explain_counterfactual_bipolar(
                trainer._bipolar, x_bin, idx, target,
                y_true=int(splits.y_test[idx]),
                feature_names=splits.feature_names,
                class_names=splits.class_names,
            )
        else:
            example_cf = explain_counterfactual_continuous(
                trainer, splits.x_test, idx, target, y_true=int(splits.y_test[idx])
            )

    row = {
        "dataset": spec.name,
        "model": "TabResNetDLBAC",
        "mode": result.mode,
        "num_features": int(splits.x_train.shape[1]),
        "num_classes": splits.num_classes,
        "best_val_accuracy": result.best_val_accuracy,
        "best_val_auroc": result.best_val_auroc,
        "training_history": result.history,
        "test_accuracy": float(metrics["accuracy"]),
        "test_auroc": metrics.get("auroc_ovr"),
        "cf_validity": float(metrics.get("counterfactuals", {}).get("validity_cf", 0.0)),
        "n_rules_exported": len(rules),
        "rules_top5": rules_top5,
        "example_rule_text": format_rule(rules[0]) if rules else None,
        "example_counterfactual": example_cf,
    }

    is_amazon = spec.name.startswith("amazon")
    threshold = 0.72 if is_amazon else 0.88
    row["acceptable"] = row["test_auroc"] is not None and float(row["test_auroc"]) >= threshold
    row["accept_threshold"] = threshold

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        ckpt = save_dir / f"{spec.name}_model.pt"
        trainer.save_checkpoint(ckpt)
        row["model_checkpoint"] = str(ckpt)
        (save_dir / f"{spec.name}_config.json").write_text(
            json.dumps(asdict(cfg), indent=2), encoding="utf-8"
        )
        (save_dir / f"{spec.name}_history.json").write_text(
            json.dumps(result.history, indent=2), encoding="utf-8"
        )
        row["history_file"] = str(save_dir / f"{spec.name}_history.json")

    return row


def main() -> None:
    p = argparse.ArgumentParser(description="TabResNet DLBAC : regles + contrefactuels")
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
                json.dumps(row, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
            )
        flag = "PASS" if row["acceptable"] else "FAIL"
        auroc = row["test_auroc"]
        print(
            f"\n  [{flag}] {name} mode={row['mode']} auroc={auroc:.4f} acc={row['test_accuracy']:.4f} "
            f"cf_valid={row['cf_validity']:.4f} rules={row['n_rules_exported']}",
            flush=True,
        )
        if row.get("example_rule_text"):
            print(f"  Regle: {row['example_rule_text']}", flush=True)

    if args.save:
        (RESULTS_DIR / "summary.json").write_text(
            json.dumps(rows, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
