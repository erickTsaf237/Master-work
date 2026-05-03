from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HybridDRConfig:
    seed: int = 42
    epochs: int = 80
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-5

    num_rules: int = 48
    hyper_hidden_dim: int = 128
    cf_hidden_dim: int = 128
    # τ dans o_k = exp(-u_k^2/τ) (porte HyperLogic) ; plus petit = gating plus net.
    temperature: float = 0.15

    bins_per_feature: int = 4

    cf_lambda: float = 0.35
    flip_lambda: float = 0.06
    rule_sparsity_lambda: float = 0.002
    grad_clip_norm: float = 1.0
