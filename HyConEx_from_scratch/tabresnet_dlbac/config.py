from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TabResNetDLBACConfig:
    """Config unifiee DLBAC : TabResNet instance (basse dim) ou embed+2 phases (haute dim)."""

    seed: int = 42
    max_instance_dim: int = 512

    # Basse dimension — HybridDRNetModel (TabResNet par echantillon)
    instance_epochs: int = 40
    instance_batch_size: int = 128
    instance_lr: float = 1e-3
    instance_num_rules: int = 64
    instance_temperature: float = 0.7
    instance_cf_lambda: float = 0.15
    instance_flip_lambda: float = 0.05
    instance_cf_warmup: int = 3

    # Haute dimension — PureDRNet embed + greffe CF
    embed_epochs: int = 40
    embed_cf_epochs: int = 12
    embed_batch_size: int = 96
    embed_lr: float = 1e-3
    embed_num_rules: int = 48
    embed_temperature: float = 1.0
    embed_dim: int = 256
    embed_cf_lambda: float = 0.08
    embed_flip_lambda: float = 0.04
    embed_distill_lambda: float = 1.5

    hyper_hidden_dim: int = 128
    cf_hidden_dim: int = 128
    tabresnet_n_blocks: int = 4
    tabresnet_dropout: float = 0.1
    rule_sparsity_lambda: float = 0.002
    input_encoding: str = "auto"
    early_stop_metric: str = "auto"
    use_class_weights: bool = True
    min_auroc_for_cf_phase2: float = 0.50
