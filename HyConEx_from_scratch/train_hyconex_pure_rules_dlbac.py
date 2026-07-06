"""
HyConEx pur + hypernet local + tete DR-Net (regles sur z) sur DLBAC.

Explications : regles z_*, hypernet local, pont oh_*, contrefactuels.

Usage :
    python train_hyconex_pure_rules_dlbac.py --dataset u4k-r4k-auth11k amazon1 --save
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from hyconex_pure_rules import (
    HyConExLocalRulesTrainer,
    RulesConfig,
    decode_rule_for_sample,
    decode_rules_batch,
    explain_counterfactual,
    explain_input_bridge,
    explain_local_hypernet,
    format_decoded_rule,
    format_rule,
)
from prepare_dlbac_datasets import discover_dlbac_datasets
from train_nouveau_module_dlbac_quantile import build_onehot_splits

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results" / "hyconex_pure_rules_dlbac"


def discover_specs():
    return [s for s in discover_dlbac_datasets() if s.has_train]


def config_for_dataset(name: str, n_features: int, num_classes: int) -> RulesConfig:
    is_amazon = name.startswith("amazon")
    high = n_features > 1000
    return RulesConfig(
        seed=42,
        epochs=25 if is_amazon else 35,
        batch_size=32 if high else 128,
        lr=8e-4 if high else 1e-3,
        latent_dim=64 if high else 32,
        hidden_dim=128 if high else 64,
        cf_lambda=0.35 if not high else 0.25,
        l1_lambda=0.02 if high else 0.01,
        l2_lambda=0.005,
        num_rules=48 if high else 64,
        temperature=0.5 if high else 0.6,
        hyper_weight=0.78,
        rule_weight=0.22,
        rule_sparsity_lambda=0.003,
    )


def batched_predict_proba(trainer, x: np.ndarray, *, batch_size: int = 256) -> np.ndarray:
    trainer.model.eval()
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            xb = torch.tensor(x[start : start + batch_size], dtype=torch.float32, device=trainer.device)
            proba = torch.softmax(trainer.model(xb), dim=1).cpu().numpy()
            chunks.append(proba)
    return np.vstack(chunks)


def batched_accuracy(trainer, x: np.ndarray, y: np.ndarray, *, batch_size: int = 256) -> float:
    proba = batched_predict_proba(trainer, x, batch_size=batch_size)
    return float((np.argmax(proba, axis=1) == y).mean())


def pick_cf_example(trainer, x_test, y_test, class_names, feature_names, max_probe: int = 32):
    for i in range(min(max_probe, len(y_test))):
        with torch.no_grad():
            x_t = torch.tensor(x_test[i : i + 1], dtype=torch.float32, device=trainer.device)
            pred = int(trainer.model(x_t).argmax().item())
        for target in range(len(class_names)):
            if target == pred:
                continue
            cf = explain_counterfactual(
                trainer, x_test, i, target,
                feature_names=feature_names, class_names=class_names,
                y_true=int(y_test[i]),
            )
            if cf["valid"]:
                return i, target
    return None


def train_one(spec, *, save_dir: Path | None = None, verbose: bool = True) -> dict:
    splits = build_onehot_splits(spec, val_size=0.2, random_state=42, use_cache=True)
    cfg = config_for_dataset(spec.name, splits.x_train.shape[1], splits.num_classes)
    trainer = HyConExLocalRulesTrainer(cfg)
    trainer.class_names = splits.class_names
    trainer.feature_names = splits.feature_names

    if verbose:
        print(f"\n=== {spec.name} (HyConEx + regles sur z) ===", flush=True)
        print(
            f"  features: {splits.x_train.shape[1]} | classes: {splits.num_classes} | "
            f"hyper={cfg.hyper_weight} rule={cfg.rule_weight} K={cfg.num_rules}",
            flush=True,
        )
        print(f"  Device: {trainer.device}", flush=True)

    result = trainer.fit(
        splits.x_train, splits.y_train,
        X_val=splits.x_val, y_val=splits.y_val,
        verbose=verbose,
    )

    high_dim = splits.x_train.shape[1] > 1000
    cf_samples = 512 if high_dim else 4000
    eval_bs = 64 if high_dim else 256
    metrics = trainer.evaluate(
        splits.x_test, splits.y_test, counterfactuals=True, cf_max_samples=cf_samples,
    )
    if high_dim:
        metrics["accuracy"] = batched_accuracy(trainer, splits.x_test, splits.y_test, batch_size=eval_bs)

    proba = batched_predict_proba(trainer, splits.x_test, batch_size=eval_bs)
    if splits.num_classes == 2:
        test_auroc = float(roc_auc_score(splits.y_test, proba[:, 1]))
    else:
        test_auroc = float(metrics.get("auroc_ovr") or roc_auc_score(splits.y_test, proba, multi_class="ovr"))

    rules = trainer.export_rules(top_per_rule=4, min_abs_weight=0.001)
    rules_top5 = [
        {"text": format_rule(r), "score": r["score"], "then_class": r["then_class"]}
        for r in rules[:5]
    ]

    max_probe = 8 if high_dim else 32
    picked = pick_cf_example(
        trainer, splits.x_test, splits.y_test, splits.class_names, splits.feature_names, max_probe=max_probe,
    )
    if picked is not None:
        idx_ex, target = picked
    else:
        idx_ex = 0
        target = 1 if splits.num_classes > 1 and int(splits.y_test[0]) == 0 else 0

    example_local = explain_local_hypernet(
        trainer, splits.x_test, idx_ex, class_names=splits.class_names, y_true=int(splits.y_test[idx_ex]),
    )
    example_bridge = explain_input_bridge(
        trainer, splits.x_test, idx_ex,
        feature_names=splits.feature_names, class_names=splits.class_names,
    )
    example_cf = explain_counterfactual(
        trainer, splits.x_test, idx_ex, target,
        feature_names=splits.feature_names, class_names=splits.class_names,
        y_true=int(splits.y_test[idx_ex]),
    )
    example_rule = rules[0] if rules else None
    example_rule_decoded = None
    rules_top5_decoded: list[dict] = []
    if rules:
        example_rule_decoded = decode_rule_for_sample(
            trainer,
            example_rule,
            splits.x_test,
            idx_ex,
            feature_names=splits.feature_names,
        )
        rules_top5_decoded = decode_rules_batch(
            trainer,
            rules,
            splits.x_test,
            idx_ex,
            feature_names=splits.feature_names,
            max_rules=5,
        )

    model_path = None
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        model_path = save_dir / f"{spec.name}_model.pt"
        torch.save(
            {
                "state_dict": trainer.model.state_dict(),
                "config": asdict(cfg),
                "class_names": splits.class_names,
                "feature_names": splits.feature_names,
                "input_dim": splits.x_train.shape[1],
                "num_classes": splits.num_classes,
                "model_type": "HyConExLocalRulesModel",
            },
            model_path,
        )

    is_amazon = spec.name.startswith("amazon")
    threshold = 0.72 if is_amazon else 0.88

    return {
        "dataset": spec.name,
        "model": "HyConExLocalRulesModel",
        "hyper_weight": cfg.hyper_weight,
        "rule_weight": cfg.rule_weight,
        "num_rules": cfg.num_rules,
        "num_features": int(splits.x_train.shape[1]),
        "num_classes": splits.num_classes,
        "latent_dim": cfg.latent_dim,
        "best_val_accuracy": float(result.best_val_accuracy),
        "test_accuracy": float(metrics["accuracy"]),
        "test_auroc": test_auroc,
        "acceptable": test_auroc >= threshold,
        "accept_threshold": threshold,
        "cf_validity": float(metrics.get("counterfactuals", {}).get("validity_cf", 0.0)),
        "cf_proximity_l1": float(metrics.get("counterfactuals", {}).get("proximity_l1_mean", 0.0)),
        "n_rules_exported": len(rules),
        "rules_top5": rules_top5,
        "example_rule": example_rule,
        "example_rule_text": format_rule(example_rule) if example_rule else None,
        "example_rule_decoded": example_rule_decoded,
        "example_rule_decoded_text": format_decoded_rule(example_rule_decoded) if example_rule_decoded else None,
        "rules_top5_decoded": rules_top5_decoded,
        "example_local_hypernet": example_local,
        "example_input_bridge": example_bridge,
        "example_counterfactual": example_cf,
        "model_checkpoint": str(model_path) if model_path else None,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="HyConEx pur + regles DR-Net sur DLBAC")
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
                json.dumps(row, indent=2, ensure_ascii=False, default=str), encoding="utf-8",
            )
        flag = "PASS" if row["acceptable"] else "FAIL"
        print(
            f"\n  [{flag}] {name} auroc={row['test_auroc']:.4f} acc={row['test_accuracy']:.4f} "
            f"rules={row['n_rules_exported']} cf_valid={row['cf_validity']:.4f}",
            flush=True,
        )
        if row.get("example_rule_decoded_text"):
            print(row["example_rule_decoded_text"], flush=True)

    if args.save:
        (RESULTS_DIR / "summary.json").write_text(
            json.dumps(rows, indent=2, ensure_ascii=False, default=str), encoding="utf-8",
        )


if __name__ == "__main__":
    main()
