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
    # TabResNet (hyperréseau) : largeur cachée = hyper_hidden_dim pour compat scripts existants.
    hyper_hidden_dim: int = 128
    tabresnet_n_blocks: int = 4
    tabresnet_dropout: float = 0.1
    cf_hidden_dim: int = 128
    # τ dans o_k = exp(-u_k^2/τ). Trop petit → activations quasi nulles, peu d’apprentissage.
    temperature: float = 0.8

    bins_per_feature: int = 4
    # auto : bipolar si entrée déjà {0,1}/{-1,+1}, sinon quantiles (Iris, etc.)
    input_encoding: str = "auto"
    use_class_weights: bool = True
    # early stopping : auto (auroc si déséquilibre), accuracy ou auroc
    early_stop_metric: str = "auto"
    # Premières époques : classification seule (cf_lambda/flip_lambda forcés à 0)
    cf_warmup_epochs: int = 0

    cf_lambda: float = 0.18
    flip_lambda: float = 0.06
    rule_sparsity_lambda: float = 0.002
    grad_clip_norm: float = 1.0

    use_focal_loss: bool = False
    focal_gamma: float = 2.0
    use_weighted_sampler: bool = False
