from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RuleConExConfig:
    seed: int = 42
    epochs: int = 40
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    eval_batch_size: int = 256

    # Architecture
    latent_dim: int = 64
    hidden_dim: int = 128
    num_rules: int = 48
    temperature: float = 0.7
    tabresnet_blocks: int = 3
    dropout: float = 0.1
    use_deep_branch: bool = True
    deep_weight: float = 0.15

    # Fusion branches (somme pondérée, normalisée en softmax dans le modèle)
    hyconex_weight: float = 0.45
    rules_weight: float = 0.45

    # Monte Carlo HyperLogic : M1 train, M2 inference
    mc_train_samples: int = 3
    mc_infer_samples: int = 5

    # Pertes
    cf_lambda: float = 0.12
    flip_lambda: float = 0.04
    conex_lambda: float = 0.08
    rule_sparsity_lambda: float = 0.002
    kl_diversity_lambda: float = 0.05
    importance_l1_lambda: float = 0.01

    early_stop_metric: str = "auto"  # auto | accuracy | auroc | deny_f1
    use_class_weights: bool = True
