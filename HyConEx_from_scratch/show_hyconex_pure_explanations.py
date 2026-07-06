"""Affiche prédictions et explications HyConEx pur (CF + saliency) depuis un checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from hyconex_from_scratch import HyConExTrainer, TrainConfig
from prepare_dlbac_datasets import discover_dlbac_datasets
from train_hyconex_pure_dlbac import (
    RESULTS_DIR,
    explain_counterfactual,
    explain_gradient_saliency,
)
from train_nouveau_module_dlbac_quantile import build_onehot_splits

ROOT = Path(__file__).resolve().parent


def load_trainer(ckpt_path: Path) -> tuple[HyConExTrainer, dict]:
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = TrainConfig(**payload["config"])
    trainer = HyConExTrainer(cfg)
    trainer._ensure_model(payload["input_dim"], payload["num_classes"])
    trainer.model.load_state_dict(payload["state_dict"])
    trainer.model.to(trainer.device)
    trainer.model.eval()
    return trainer, payload


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="u4k-r4k-auth11k")
    p.add_argument("--sample-idx", type=int, default=0)
    p.add_argument("--target", type=int, default=None, help="Classe cible CF (défaut: autre que prédite)")
    args = p.parse_args()

    results_path = RESULTS_DIR / f"{args.dataset}_results.json"
    ckpt_path = RESULTS_DIR / f"{args.dataset}_model.pt"
    if not ckpt_path.exists():
        raise SystemExit(f"Checkpoint introuvable: {ckpt_path}")

    specs = {s.name: s for s in discover_dlbac_datasets() if s.has_train}
    splits = build_onehot_splits(specs[args.dataset], val_size=0.2, random_state=42, use_cache=True)
    trainer, payload = load_trainer(ckpt_path)
    class_names = payload["class_names"]
    feature_names = payload["feature_names"]

    idx = args.sample_idx
    with torch.no_grad():
        x_t = torch.tensor(splits.x_test[idx : idx + 1], dtype=torch.float32, device=trainer.device)
        pred = int(trainer.model(x_t).argmax().item())
    target = args.target if args.target is not None else (1 - pred if len(class_names) == 2 else (pred + 1) % len(class_names))

    cf = explain_counterfactual(
        trainer, splits.x_test, idx, target,
        feature_names=feature_names, class_names=class_names,
        y_true=int(splits.y_test[idx]),
    )
    sal = explain_gradient_saliency(
        trainer, splits.x_test, idx,
        feature_names=feature_names, class_names=class_names,
    )

    print(f"\n=== HyConEx PUR — {args.dataset} — échantillon {idx} ===")
    print(f"Vrai: {class_names[int(splits.y_test[idx])]} | Prédit: {class_names[pred]}")
    print(f"\n--- Contrefactuel vers {cf['y_target_name']} ---")
    print(f"  Valide: {cf['valid']} | {cf['n_changes']} features modifiées")
    for c in cf["changes"][:8]:
        print(f"  {c['feature']}: {c['from']:.3f} -> {c['to']:.3f} (delta={c['delta']:+.3f})")
    print(f"\n--- Saliency (top 8) ---")
    for f in sal["top_features"][:8]:
        print(f"  {f['feature']}: grad={f['gradient']:+.3f} val={f['value']:.3f}")

    if results_path.exists():
        saved = json.loads(results_path.read_text(encoding="utf-8"))
        print(f"\n--- Métriques test sauvegardées ---")
        print(f"  AUROC={saved['test_auroc']:.4f} acc={saved['test_accuracy']:.4f} cf_valid={saved['cf_validity']:.4f}")


if __name__ == "__main__":
    main()
