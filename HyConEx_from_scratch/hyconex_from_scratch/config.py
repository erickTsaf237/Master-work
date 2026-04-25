from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrainConfig:
    """Hyperparamètres d'entraînement HyConEx from-scratch."""

    seed: int = 42
    epochs: int = 100
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-5
    latent_dim: int = 32
    hidden_dim: int = 64
    cf_lambda: float = 0.35
    l1_lambda: float = 0.01
    l2_lambda: float = 0.005
