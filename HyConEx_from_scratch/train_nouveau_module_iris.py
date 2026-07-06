"""Exemple minimal: entraînement Iris avec le nouveau module DR-HyperCF binaire."""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from nouveau_module import HybridDRConfig, HybridDRTrainer


def main() -> None:
    iris = load_iris()
    x_raw = iris.data.astype(np.float32)
    y = iris.target.astype(np.int64)

    x_train_raw, x_test_raw, y_train, y_test = train_test_split(
        x_raw, y, test_size=0.3, random_state=42, stratify=y
    )
    x_train_raw, x_val_raw, y_train, y_val = train_test_split(
        x_train_raw, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train_raw).astype(np.float32)
    x_val = scaler.transform(x_val_raw).astype(np.float32)
    x_test = scaler.transform(x_test_raw).astype(np.float32)

    cfg = HybridDRConfig(
        epochs=40,
        batch_size=16,
        num_rules=24,
        bins_per_feature=4,
        temperature=0.8,
        cf_lambda=0.15,
        flip_lambda=0.06,
    )
    trainer = HybridDRTrainer(cfg)
    result = trainer.fit(
        x_train,
        y_train,
        x_val_cont=x_val,
        y_val=y_val,
        feature_names=iris.feature_names,
        class_names=iris.target_names.tolist(),
        verbose=True,
    )

    metrics = trainer.evaluate(x_test, y_test, counterfactuals=True)
    rules = trainer.export_rules(top_per_rule=3)

    print("\n=== Nouveau module DR-HyperCF ===")
    print(f"Best val accuracy: {result.best_val_accuracy:.4f}")
    print(f"Test accuracy: {metrics['accuracy']:.4f}")
    print(f"CF validity: {metrics['counterfactuals']['validity_cf']:.4f}")
    print(f"Changed bits mean: {metrics['counterfactuals']['changed_bits_mean']:.4f}")
    print(f"Nb rules extraites: {len(rules)}")
    print("Exemple règle:", rules[0] if rules else "aucune")


if __name__ == "__main__":
    main()
