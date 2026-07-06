"""
Test rapide RuleConEx (exécuter depuis HyConEx_from_scratch).

    python ruleconex/test_ruleconex.py
    python ruleconex/test_ruleconex.py --dataset amazon1 --epochs 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prepare_dlbac_datasets import discover_dlbac_datasets
from ruleconex.config import RuleConExConfig
from ruleconex.evaluate import evaluate_counterfactuals, evaluate_ruleconex
from ruleconex.loss import ruleconex_loss
from ruleconex.model import RuleConExModel
from ruleconex.trainer import RuleConExTrainer
from ruleconex.utils import explain_sample, extract_rules_from_pack
from train_nouveau_module_dlbac_quantile import build_onehot_splits

import torch


def test_forward_pass(input_dim: int = 32, num_classes: int = 2) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RuleConExModel(input_dim, num_classes, num_rules=8, mc_train_samples=2, mc_infer_samples=3).to(device)
    x = torch.rand(4, input_dim, device=device)
    pack = model.forward_pack(x)
    assert pack.logits.shape == (4, num_classes)
    assert pack.input_importance.shape == (4, num_classes, input_dim)
    assert len(pack.mc_logits_rules) == 2
    y_t = torch.tensor([1, 0, 1, 0], device=device)
    x_cf = model.generate_counterfactual(x, y_t, pack=pack)
    assert x_cf.shape == x.shape
    print("[OK] forward + counterfactual")


def test_loss() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RuleConExModel(20, 2, num_rules=6, mc_train_samples=2).to(device)
    cfg = RuleConExConfig(cf_lambda=0.1, mc_train_samples=2)
    x = torch.rand(8, 20, device=device)
    y = torch.randint(0, 2, (8,), device=device)
    model.train()
    pack = model.forward_pack(x, mc_samples=2)
    bd = ruleconex_loss(model, pack, y, x, cfg)
    bd.total.backward()
    print(f"[OK] loss backward | total={bd.total.item():.4f}")


def test_training(dataset: str, epochs: int) -> None:
    specs = {s.name: s for s in discover_dlbac_datasets()}
    if dataset not in specs:
        raise SystemExit(f"Dataset {dataset} introuvable")

    splits = build_onehot_splits(specs[dataset], random_state=42)
    cfg = RuleConExConfig(epochs=epochs, batch_size=128, mc_train_samples=2, mc_infer_samples=3)
    if dataset.startswith("amazon"):
        cfg.epochs = min(epochs, 8)

    trainer = RuleConExTrainer(cfg)
    trainer.fit(
        splits.x_train,
        splits.y_train,
        splits.x_val,
        splits.y_val,
        feature_names=splits.feature_names,
        class_names=splits.class_names,
        verbose=True,
    )

    ev = evaluate_ruleconex(trainer, splits.x_test, splits.y_test)
    print(f"[OK] test accuracy={ev['metrics']['accuracy']:.4f}")
    cf = ev["counterfactuals"]
    print(
        f"[OK] CF validity={cf['validity_cf']:.3f} | "
        f"features modifiées={cf['changed_features_mean']:.1f} | "
        f"L1={cf['proximity_l1_mean']:.2f} | "
        f"succès/échantillon={cf['flip_success_rate']:.3f}"
    )

    rep = explain_sample(
        trainer.model,
        splits.x_test[0],
        None,
        feature_names=splits.feature_names,
        class_names=splits.class_names,
        device=trainer.device,
    )
    print(rep.text_report[:500], "...")

    model = trainer.model
    model.eval()
    with torch.no_grad():
        pack = model.forward_pack(torch.tensor(splits.x_test[:1], dtype=torch.float32, device=trainer.device))
    rules = extract_rules_from_pack(
        pack,
        splits.feature_names,
        splits.class_names,
        rules_on_input=model.rules_on_input,
        latent_dim=model.latent_dim,
    )
    print(f"[OK] {len(rules)} règles extraites")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="u4k-r4k-auth11k")
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--unit-only", action="store_true")
    args = p.parse_args()

    test_forward_pass()
    test_loss()
    if not args.unit_only:
        if not torch.cuda.is_available():
            raise SystemExit("CUDA requis pour les tests RuleConEx.")
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
        test_training(args.dataset, args.epochs)


if __name__ == "__main__":
    main()
