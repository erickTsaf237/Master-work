"""Affiche règle + contrefactuel pour un DR-Net pur sauvegardé."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hyperlogic_pure import PureDRNetTrainer
from prepare_dlbac_datasets import explain_counterfactual_continuous, format_rule
from train_nouveau_module_dlbac_quantile import build_onehot_splits
from prepare_dlbac_datasets import discover_dlbac_datasets

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results" / "pure_drnet_dlbac"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="u4k-r4k-auth11k")
    args = p.parse_args()

    ckpt = RESULTS_DIR / f"{args.dataset}_model.pt"
    if not ckpt.exists():
        raise SystemExit(f"Checkpoint introuvable: {ckpt}")

    trainer = PureDRNetTrainer.load_checkpoint(ckpt)
    print(f"Modele: {ckpt} | mode={trainer.model.mode if trainer.model else '?'}")

    json_path = RESULTS_DIR / f"{args.dataset}_results.json"
    if json_path.exists():
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        if payload.get("example_rule"):
            print("\n=== REGLE ===")
            print(format_rule(payload["example_rule"]))
        if payload.get("example_counterfactual"):
            cf = payload["example_counterfactual"]
            print("\n=== CONTREFACTUEL ===")
            print(f"#{cf['sample_idx']}: {cf['y_pred_orig_name']} -> {cf['y_pred_cf_name']} valid={cf['valid']}")
            for ch in cf.get("changes", [])[:8]:
                print(f"  {ch['feature']}: {ch['from']:.3f} -> {ch['to']:.3f}")
        return

    rules = trainer.export_rules(top_per_rule=4, min_abs_weight=0.001)
    if rules:
        print("\n=== REGLE ===")
        print(format_rule(rules[0]))


if __name__ == "__main__":
    main()
