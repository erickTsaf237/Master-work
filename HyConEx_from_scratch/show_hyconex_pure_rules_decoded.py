"""Affiche regles decodees (z -> oh_*) depuis un checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from hyconex_pure_rules import (
    HyConExLocalRulesTrainer,
    decode_rule_for_sample,
    format_decoded_rule,
    format_rule,
)
from hyconex_pure_rules.config import RulesConfig
from prepare_dlbac_datasets import discover_dlbac_datasets
from train_hyconex_pure_rules_dlbac import RESULTS_DIR
from train_nouveau_module_dlbac_quantile import build_onehot_splits


def load_trainer(ckpt_path: Path) -> HyConExLocalRulesTrainer:
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = RulesConfig(**payload["config"])
    trainer = HyConExLocalRulesTrainer(cfg)
    trainer._ensure_model(payload["input_dim"], payload["num_classes"])
    trainer.model.load_state_dict(payload["state_dict"])
    trainer.model.to(trainer.device)
    trainer.model.eval()
    trainer.class_names = list(payload["class_names"])
    trainer.feature_names = list(payload["feature_names"])
    return trainer


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="amazon1")
    p.add_argument("--sample-idx", type=int, default=0)
    p.add_argument("--rule-rank", type=int, default=0)
    args = p.parse_args()

    ckpt_path = RESULTS_DIR / f"{args.dataset}_model.pt"
    results_path = RESULTS_DIR / f"{args.dataset}_results.json"
    if not ckpt_path.exists():
        raise SystemExit(f"Checkpoint introuvable: {ckpt_path}")

    trainer = load_trainer(ckpt_path)
    specs = {s.name: s for s in discover_dlbac_datasets() if s.has_train}
    splits = build_onehot_splits(specs[args.dataset], val_size=0.2, random_state=42, use_cache=True)

    rules = trainer.export_rules(top_per_rule=4, min_abs_weight=0.001)
    rule = rules[args.rule_rank]

    decoded = decode_rule_for_sample(
        trainer, rule, splits.x_test, args.sample_idx, feature_names=splits.feature_names,
    )

    print(f"\n=== Regle decodee — {args.dataset} — echantillon {args.sample_idx} ===")
    print(format_rule(rule))
    print(format_decoded_rule(decoded))

    if results_path.exists():
        saved = json.loads(results_path.read_text(encoding="utf-8"))
        print(f"\nMetriques: AUROC={saved.get('test_auroc', 0):.4f} acc={saved.get('test_accuracy', 0):.4f}")


if __name__ == "__main__":
    main()
