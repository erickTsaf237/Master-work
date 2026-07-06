"""
CLI RuleConEx sur jeux DLBAC.

Usage :
    python -m ruleconex.main --dataset u4k-r4k-auth11k
    python -m ruleconex.main --dataset amazon1 --epochs 30
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from prepare_dlbac_datasets import discover_dlbac_datasets
from ruleconex.config import RuleConExConfig
from ruleconex.evaluate import evaluate_ruleconex, run_all_baselines
from ruleconex.trainer import RuleConExTrainer
from ruleconex.utils import explain_sample
from train_nouveau_module_dlbac_quantile import build_onehot_splits


ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Entraînement RuleConEx DLBAC")
    p.add_argument("--dataset", type=str, default="u4k-r4k-auth11k")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-baselines", action="store_true")
    p.add_argument("--explain", action="store_true", help="Affiche un exemple d'explication")
    p.add_argument("--out-dir", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    specs = {s.name: s for s in discover_dlbac_datasets()}
    if args.dataset not in specs:
        raise SystemExit(f"Dataset inconnu : {args.dataset}. Disponibles : {sorted(specs)}")

    splits = build_onehot_splits(specs[args.dataset], random_state=args.seed)
    cfg = RuleConExConfig(seed=args.seed)
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.lr is not None:
        cfg.lr = args.lr

    if splits.name.startswith("amazon"):
        cfg.epochs = min(cfg.epochs, 35)
        cfg.num_rules = 48

    if not torch.cuda.is_available():
        raise SystemExit("CUDA requis pour RuleConEx. Aucun GPU détecté.")

    print(f"=== RuleConEx | {splits.name} | {splits.x_train.shape[1]} features | {splits.num_classes} classes ===")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    trainer = RuleConExTrainer(cfg)
    result = trainer.fit(
        splits.x_train,
        splits.y_train,
        splits.x_val,
        splits.y_val,
        feature_names=splits.feature_names,
        class_names=splits.class_names,
        verbose=True,
    )

    eval_out = evaluate_ruleconex(trainer, splits.x_test, splits.y_test)
    print("\n--- Test RuleConEx ---")
    for k, v in eval_out["metrics"].items():
        print(f"  {k}: {v:.4f}")

    baseline_results = []
    if not args.no_baselines:
        print("\n--- Baselines ---")
        baseline_results = run_all_baselines(
            splits.x_train,
            splits.y_train,
            splits.x_val,
            splits.y_val,
            splits.x_test,
            splits.y_test,
            epochs=min(25, cfg.epochs),
            seed=args.seed,
        )
        for br in baseline_results:
            if "error" in br:
                print(f"  {br['name']}: ERREUR — {br['error']}")
            else:
                m = br["metrics"]
                print(f"  {br['name']}: acc={m['accuracy']:.4f} f1={m['f1_macro']:.4f}")

    if args.explain:
        rep = explain_sample(
            trainer.model,
            splits.x_test[0],
            None,
            feature_names=splits.feature_names,
            class_names=splits.class_names,
            device=trainer.device,
        )
        print("\n" + rep.text_report)

    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "outputs" / "ruleconex" / splits.name
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": splits.name,
        "config": cfg.__dict__,
        "history": result.history,
        "test_metrics": eval_out["metrics"],
        "baselines": [
            {k: v for k, v in br.items() if k != "y_pred" and k != "y_proba"}
            for br in baseline_results
        ],
    }
    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nRésultats sauvegardés : {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
