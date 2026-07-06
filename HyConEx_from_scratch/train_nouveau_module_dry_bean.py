from __future__ import annotations

import numpy as np

from feature_engineering_dry_bean import prepare_dry_bean_splits
from nouveau_module import HybridDRConfig, HybridDRTrainer


def main() -> None:
    splits = prepare_dry_bean_splits(
        test_size=0.2,
        val_size=0.2,
        random_state=42,
        add_engineered_features=True,
        clip_outliers=True,
    )

    cfg = HybridDRConfig(
        seed=42,
        epochs=40,
        batch_size=128,
        lr=1e-3,
        num_rules=64,
        bins_per_feature=4,
        temperature=0.8,
        cf_lambda=0.15,
        flip_lambda=0.06,
        rule_sparsity_lambda=0.002,
    )

    trainer = HybridDRTrainer(cfg)
    result = trainer.fit(
        splits.X_train,
        splits.y_train,
        x_val_cont=splits.X_val,
        y_val=splits.y_val,
        feature_names=splits.feature_names,
        class_names=splits.class_names,
        verbose=True,
    )

    metrics = trainer.evaluate(splits.X_test, splits.y_test, counterfactuals=True)
    rules = trainer.export_rules(top_per_rule=4)

    print("\n=== Nouveau module DR-HyperCF sur Dry Bean ===")
    print(f"Best val accuracy: {result.best_val_accuracy:.4f}")
    print(f"Test accuracy: {metrics['accuracy']:.4f}")
    print(f"Test AUROC OvR: {metrics.get('auroc_ovr')}")
    print(f"CF validity: {metrics['counterfactuals']['validity_cf']:.4f}")
    print(f"Changed bits mean: {metrics['counterfactuals']['changed_bits_mean']:.4f}")
    print(f"Proximity L1 cont mean: {metrics['counterfactuals']['proximity_l1_cont_mean']:.4f}")
    print(f"Nb rules extraites: {len(rules)}")
    if rules:
        print("Exemple règle:", rules[0])


if __name__ == "__main__":
    main()
