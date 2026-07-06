from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HybridConfig:
    seed: int = 42
    epochs: int = 40
    batch_size: int = 256
    lr: float = 3e-3
    weight_decay: float = 1e-4

    embed_dim: int = 256
    cf_hidden_dim: int = 128
    num_rules: int = 32
    temperature: float = 0.5

    linear_weight: float = 0.65
    rule_weight: float = 0.35
    cf_lambda: float = 0.0
    flip_lambda: float = 0.0
    rule_sparsity_lambda: float = 0.001

    use_class_weights: bool = True
    early_stop_metric: str = "auroc"
    grad_clip_norm: float = 1.0

    # Phase 2 (contrefactuels) — activee apres bonne classification
    cf_epochs: int = 10
    cf_lambda_phase2: float = 0.08
    flip_lambda_phase2: float = 0.03
