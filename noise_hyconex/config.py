"""Hyperparamètres Noise-only HyConEx."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class NoiseHyConExConfig:
    nr_features: int
    nr_classes: int
    noise_dim: int = 64
    hidden_size: int = 512
    n_res_blocks: int = 3
    dropout_rate: float = 0.25
    # True = contre-factuel par projection (HyConEx use_distance=True) ; False = x - w
    use_projection: bool = False
    cat_softmax_temperature: float = 0.01
    log_prob_threshold: float = -20.0
    # Perte contre-factuelle (ramps style HyConEx)
    class_lambda: float = 1.0
    dist_lambda: float = 1.0
    flow_lambda: float = 1.0
    class_start_epoch: int = 0
    dist_start_epoch: int = 0
    flow_start_epoch: int = 0
    class_warm_up_epochs: int = 50
    dist_warm_up_epochs: int = 50
    flow_warm_up_epochs: int = 50
    # Prétrain (cluster optionnel)
    cluster_lambda: float = 0.0
    cluster_start_epoch: int = 0
    # Phases
    pretrain_epochs: int = 20
    finetune_epochs: int = 50
    freeze_generator_during_pretrain: bool = False
