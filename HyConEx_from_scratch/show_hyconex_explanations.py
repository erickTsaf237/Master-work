"""
Affiche une regle DR-Net et un contrefactuel pour un modele HyConEx+HyperLogic sauvegarde.

Usage :
    python show_hyconex_explanations.py --dataset amazon1
    python show_hyconex_explanations.py --dataset amazon1 --train   # re-entraine si absent
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hyconex_hyperlogic import HyConExHyperLogicTrainer
from prepare_dlbac_datasets import (
    discover_dlbac_datasets,
    explain_counterfactual_continuous,
    format_rule,
    pick_counterfactual_example,
)
from train_nouveau_module_dlbac_quantile import build_onehot_splits

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results" / "hyconex_hyperlogic_dlbac"


def _print_rule(rule: dict) -> None:
    print("\n=== REGLE (DR-Net, espace encode emb_*) ===")
    print(format_rule(rule))
    print("Conditions detaillees :")
    for cond in rule.get("if", []):
        print(f"  - {cond}")


def _print_cf(cf: dict) -> None:
    print("\n=== CONTREFACTUEL (HyConEx, one-hot continu) ===")
    print(f"Echantillon test #{cf['sample_idx']}")
    if cf.get("y_true") is not None:
        print(f"  Label vrai     : {cf['y_true']}")
    print(f"  Prediction orig: {cf['y_pred_orig_name']} (proba={cf['proba_orig']:.3f})")
    print(f"  Cible CF       : {cf['y_target_name']}")
    print(f"  Prediction CF  : {cf['y_pred_cf_name']} (proba={cf['proba_cf']:.3f})")
    print(f"  Valide         : {cf['valid']}")
    print(f"  Nb changements : {cf['n_changes']}")
    print("  Top modifications :")
    for ch in cf.get("changes", []):
        print(f"    {ch['feature']}: {ch['from']:.3f} -> {ch['to']:.3f} (delta={ch['delta']:+.3f})")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="amazon1")
    p.add_argument("--train", action="store_true", help="Entrainer si le checkpoint est absent")
    args = p.parse_args()

    ckpt = RESULTS_DIR / f"{args.dataset}_model.pt"
    json_path = RESULTS_DIR / f"{args.dataset}_results.json"

    if not ckpt.exists():
        if not args.train:
            raise SystemExit(
                f"Checkpoint introuvable: {ckpt}\n"
                f"Lancez: python train_hyconex_hyperlogic_dlbac.py --dataset {args.dataset} --save\n"
                f"ou: python show_hyconex_explanations.py --dataset {args.dataset} --train"
            )
        from train_hyconex_hyperlogic_dlbac import discover_specs, train_one

        specs = {s.name: s for s in discover_specs()}
        if args.dataset not in specs:
            raise SystemExit(f"Dataset inconnu: {args.dataset}")
        train_one(specs[args.dataset], save_dir=RESULTS_DIR)

    trainer = HyConExHyperLogicTrainer.load_checkpoint(ckpt)
    print(f"Modele charge: {ckpt}")
    print(f"Device: {trainer.device}")

    if json_path.exists():
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        rule = payload.get("example_rule")
        if not rule:
            rules = trainer.export_rules(top_per_rule=4, min_abs_weight=0.001)
            rule = rules[0] if rules else None
        if rule:
            _print_rule(rule)
        if payload.get("example_counterfactual"):
            _print_cf(payload["example_counterfactual"])
        return

    specs = {s.name: s for s in discover_dlbac_datasets() if s.has_train}
    splits = build_onehot_splits(specs[args.dataset], val_size=0.2, random_state=42, use_cache=True)
    rules = trainer.export_rules(top_per_rule=4, min_abs_weight=0.03)
    if rules:
        _print_rule(rules[0])
    picked = pick_counterfactual_example(trainer, splits.x_test, splits.y_test)
    if picked is not None:
        idx, target = picked
        cf = explain_counterfactual_continuous(
            trainer, splits.x_test, idx, target, y_true=int(splits.y_test[idx])
        )
        _print_cf(cf)
    else:
        print("\nAucun contrefactuel valide trouve sur les premiers echantillons test.")


if __name__ == "__main__":
    main()
