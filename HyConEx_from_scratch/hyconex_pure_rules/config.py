from __future__ import annotations

from dataclasses import dataclass

from hyconex_from_scratch.config import TrainConfig


@dataclass
class RulesConfig(TrainConfig):
    """HyConEx pur + DR-Net sur z + contrefactuels."""

    num_rules: int = 48
    temperature: float = 0.5
    hyper_weight: float = 0.78
    rule_weight: float = 0.22
    rule_sparsity_lambda: float = 0.003
    ctx_modulation: float = 0.1
