"""
Smoke test : entraîne HyConEx from-scratch sur Iris (features dans [0, 1]).
Exécuter depuis la racine ``HyConEx_from_scratch``::

    python train_iris.py
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from hyconex_from_scratch import TrainConfig, train


def main() -> None:
    iris = load_iris()
    X_raw = iris.data.astype(np.float32)
    y = iris.target.astype(np.int64)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.3, random_state=42, stratify=y
    )
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_train_raw, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train_raw).astype(np.float32)
    X_val = scaler.transform(X_val_raw).astype(np.float32)
    X_test = scaler.transform(X_test_raw).astype(np.float32)

    cfg = TrainConfig(
        seed=42,
        epochs=80,
        batch_size=16,
        lr=2e-3,
        latent_dim=32,
        hidden_dim=64,
        cf_lambda=0.35,
        l1_lambda=0.01,
        l2_lambda=0.005,
    )

    result = train(
        X_train,
        y_train,
        config=cfg,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        verbose=True,
    )

    assert result.test_metrics is not None
    print("\nRésumé :")
    print(f"  Meilleure accuracy validation : {result.best_val_accuracy:.4f}")
    print(f"  Accuracy test                 : {result.test_metrics['accuracy']:.4f}")


if __name__ == "__main__":
    main()
