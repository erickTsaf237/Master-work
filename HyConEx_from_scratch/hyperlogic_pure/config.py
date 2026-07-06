from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PureDRConfig:
    """DR-Net pur (100 % règles) + greffe CF phase 2."""

    seed: int = 42
    epochs: int = 50
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-5

    num_rules: int = 48
    temperature: float = 0.8
    rule_sparsity_lambda: float = 0.002

    # Hyperréseau par échantillon (basse dimension, fidèle HyperLogic)
    hyper_hidden_dim: int = 128
    tabresnet_n_blocks: int = 4
    tabresnet_dropout: float = 0.1
    cf_hidden_dim: int = 128

    # Au-delà de ce seuil : DR-Net global + contexte batch (pas TabResNet/ligne)
    max_instance_dim: int = 512
    ctx_hidden_dim: int = 256
    ctx_modulation: float = 0.1

    use_class_weights: bool = True
    early_stop_metric: str = "auroc"
    grad_clip_norm: float = 1.0

    # Phase 1 : classification DR-Net seule
    cf_lambda: float = 0.0
    flip_lambda: float = 0.0

    # Phase 2 : greffe CF
    cf_epochs: int = 12
    cf_lambda_phase2: float = 0.08
    flip_lambda_phase2: float = 0.04
    lr_phase2: float = 5e-4
    freeze_drnet_phase2: bool = True

    # Haute dimension : CF HyConEx (delta continu) car CF binaire HyperLogic sur 14k est irréalisable
    use_cf_graft_high_dim: bool = True
    cf_graft_scale: float = 0.35
    embed_dim_high: int = 256

    # Distillation SVM -> DR-Net (Amazon haute dimension)
    distill_lambda: float = 0.0
    distill_only: bool = False
