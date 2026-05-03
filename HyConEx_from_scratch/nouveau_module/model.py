from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from nouveau_module.cf_head import generate_cf_binary
from nouveau_module.hypernet import HyperWeightGenerator
from nouveau_module.main_rule_net import main_logits_from_weights


@dataclass
class ForwardOutputs:
    logits: torch.Tensor
    rule_activations: torch.Tensor
    theta_main: torch.Tensor
    theta_cf: torch.Tensor


class HybridDRNetModel(nn.Module):
    """HyConEx + DR-Net (HyperLogic): entrée bipolar {-1,+1}, hyperréseau -> règles h(u) + tête CF."""

    def __init__(
        self,
        input_dim_bin: int,
        num_classes: int,
        num_rules: int,
        hyper_hidden_dim: int,
        cf_hidden_dim: int,
        temperature: float,
    ) -> None:
        super().__init__()
        self.input_dim_bin = input_dim_bin
        self.num_classes = num_classes
        self.num_rules = num_rules
        self.cf_hidden_dim = cf_hidden_dim
        self.temperature = temperature

        self.hyper = HyperWeightGenerator(
            input_dim_bin=input_dim_bin,
            num_classes=num_classes,
            num_rules=num_rules,
            cf_hidden_dim=cf_hidden_dim,
            hidden_dim=hyper_hidden_dim,
        )

    def forward(self, x_bin: torch.Tensor) -> ForwardOutputs:
        theta_main, theta_cf = self.hyper(x_bin)
        logits, rule_act, _ = main_logits_from_weights(
            x_bin,
            theta_main,
            input_dim=self.input_dim_bin,
            num_rules=self.num_rules,
            num_classes=self.num_classes,
            temperature=self.temperature,
        )
        return ForwardOutputs(logits=logits, rule_activations=rule_act, theta_main=theta_main, theta_cf=theta_cf)

    def predict_logits(self, x_bin: torch.Tensor) -> torch.Tensor:
        out = self.forward(x_bin)
        return out.logits

    def generate_counterfactual_binary(self, x_bin: torch.Tensor, y_target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        theta_main, theta_cf = self.hyper(x_bin)
        x_cf_bin = generate_cf_binary(
            x_bin,
            y_target,
            theta_cf=theta_cf,
            input_dim=self.input_dim_bin,
            num_classes=self.num_classes,
            hidden_dim=self.cf_hidden_dim,
        )
        logits_cf, _, _ = main_logits_from_weights(
            x_cf_bin,
            theta_main,
            input_dim=self.input_dim_bin,
            num_rules=self.num_rules,
            num_classes=self.num_classes,
            temperature=self.temperature,
        )
        return x_cf_bin, logits_cf

    def generate_counterfactuals_all_classes(
        self,
        x_bin: torch.Tensor,
        y_source: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Génère des contrefactuels pour toutes les classes cibles.

        Returns:
            x_cf_all: [B, C, D]
            logits_cf_all: [B, C, C]
        """
        theta_main, theta_cf = self.hyper(x_bin)
        bsz = x_bin.shape[0]
        device = x_bin.device

        x_cf_list: list[torch.Tensor] = []
        logits_list: list[torch.Tensor] = []
        for cls in range(self.num_classes):
            y_target = torch.full((bsz,), cls, dtype=torch.long, device=device)
            x_cf_bin = generate_cf_binary(
                x_bin,
                y_target,
                theta_cf=theta_cf,
                input_dim=self.input_dim_bin,
                num_classes=self.num_classes,
                hidden_dim=self.cf_hidden_dim,
            )
            logits_cf, _, _ = main_logits_from_weights(
                x_cf_bin,
                theta_main,
                input_dim=self.input_dim_bin,
                num_rules=self.num_rules,
                num_classes=self.num_classes,
                temperature=self.temperature,
            )
            x_cf_list.append(x_cf_bin)
            logits_list.append(logits_cf)

        x_cf_all = torch.stack(x_cf_list, dim=1)
        logits_cf_all = torch.stack(logits_list, dim=1)
        return x_cf_all, logits_cf_all
