from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CFTeacherDRConfig:
    """
    DR-Net TabResNet en 2 phases :
    - Phase 1 : la voie contrefactuelle fait la prédiction (CF -> règles sur x_cf)
    - Phase 2 : la tête règles apprend du teacher CF (distillation)
    """

    seed: int = 42
    phase1_epochs: int = 35
    phase2_epochs: int = 12
    batch_size: int = 128
    lr: float = 1e-3
    lr_phase2: float = 5e-4
    weight_decay: float = 1e-5

    num_rules: int = 64
    temperature: float = 0.7
    rule_sparsity_lambda: float = 0.002

    hyper_hidden_dim: int = 128
    tabresnet_n_blocks: int = 4
    tabresnet_dropout: float = 0.1
    cf_hidden_dim: int = 128

    max_instance_dim: int = 512
    ctx_hidden_dim: int = 256
    ctx_modulation: float = 0.1
    embed_dim_high: int = 128

    use_class_weights: bool = True
    early_stop_metric: str = "auroc"
    grad_clip_norm: float = 1.0

    # Phase 1 : CF prédit (pas de CE directe sur règles(x))
    cf_lambda: float = 0.35
    flip_lambda: float = 0.04
    cf_predict_lambda: float = 1.0

    # Phase 2 : règles apprennent du teacher CF
    distill_lambda: float = 1.5
    distill_temperature: float = 2.0
    rules_ce_lambda: float = 1.0
    freeze_cf_phase2: bool = True
    freeze_tab_phase2: bool = False

    bins_per_feature: int = 4
    input_encoding: str = "bipolar"
