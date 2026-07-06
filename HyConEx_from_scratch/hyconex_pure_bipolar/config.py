from __future__ import annotations

from dataclasses import dataclass

from hyconex_pure_rules.config import RulesConfig


@dataclass
class BipolarRulesConfig(RulesConfig):
    """HyConEx + DR-Net avec entree {-1,+1} (fidele HyperLogic)."""

    max_drnet_input_dim: int = 512
    flip_lambda: float = 0.02
    distill_lambda: float = 1.0
    distill_temperature: float = 2.0
    rules_phase_epochs: int = 8
    rules_phase_lr_scale: float = 0.5
