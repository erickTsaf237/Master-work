from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from hyconex_pure_local.model import HyConExLocalModel, LocalHypernetPack
from nouveau_module.main_rule_net import main_logits_from_weights


def to_bipolar(x: torch.Tensor) -> torch.Tensor:
    if x.min() >= -1.01 and x.max() <= 1.01:
        return torch.where(x > 0.25, torch.ones_like(x), -torch.ones_like(x))
    return torch.where(x > 0.5, torch.ones_like(x), -torch.ones_like(x))


def _main_theta_dim(input_dim: int, num_rules: int, num_classes: int) -> int:
    return input_dim * num_rules + num_rules + num_rules * num_classes + num_classes


@dataclass
class RulesForwardPack:
    logits: torch.Tensor
    logits_hyper: torch.Tensor
    logits_rules: torch.Tensor
    rule_activations: torch.Tensor
    hyper_pack: LocalHypernetPack
    theta_main: torch.Tensor
    rule_params: tuple[torch.Tensor, ...]


class HyConExLocalRulesModel(HyConExLocalModel):
    """
    HyConEx pur + tête DR-Net sur l'espace latent z.

    logits = hyper_weight * W(z)·z + rule_weight * DR-Net(bipolar(z))
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        latent_dim: int,
        hidden_dim: int,
        *,
        num_rules: int = 48,
        temperature: float = 0.5,
        hyper_weight: float = 0.78,
        rule_weight: float = 0.22,
        ctx_modulation: float = 0.1,
    ) -> None:
        super().__init__(input_dim, num_classes, latent_dim, hidden_dim)
        self.num_rules = num_rules
        self.temperature = temperature
        self.hyper_weight = hyper_weight
        self.rule_weight = rule_weight
        self.ctx_modulation = ctx_modulation

        theta_dim = _main_theta_dim(latent_dim, num_rules, num_classes)
        self.theta_bias = nn.Parameter(torch.zeros(theta_dim))
        self.ctx_to_theta = nn.Linear(latent_dim, theta_dim)

    def _theta_for_batch(self, z: torch.Tensor) -> torch.Tensor:
        if self.training and z.shape[0] <= 256:
            ctx = z.mean(dim=0)
            delta = self.ctx_to_theta(ctx)
            return self.theta_bias + self.ctx_modulation * delta
        return self.theta_bias

    def forward_pack(self, x: torch.Tensor) -> RulesForwardPack:
        hyper_pack = self.local_hypernet_pack(x)
        z_bin = to_bipolar(hyper_pack.z)
        theta = self._theta_for_batch(hyper_pack.z)
        logits_rules, rule_act, rule_params = main_logits_from_weights(
            z_bin,
            theta,
            input_dim=self.latent_dim,
            num_rules=self.num_rules,
            num_classes=self.num_classes,
            temperature=self.temperature,
        )
        logits = self.hyper_weight * hyper_pack.logits + self.rule_weight * logits_rules
        return RulesForwardPack(
            logits=logits,
            logits_hyper=hyper_pack.logits,
            logits_rules=logits_rules,
            rule_activations=rule_act,
            hyper_pack=hyper_pack,
            theta_main=theta,
            rule_params=rule_params,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_pack(x).logits
