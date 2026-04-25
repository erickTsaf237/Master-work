"""Dual-head model: HyConEx TabResNet hypernetwork + HyperLogic DR branch."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .hyperlogic_drnet import HyperLogicBranch
from .paths import ensure_hyconex_path

ensure_hyconex_path()
from hyconex.hypernetwork import HyperNet  # noqa: E402


class DualExplainModel(nn.Module):
    """
    Single forward pass:
      - HyConEx head: multiclass logits + local linear weights (counterfactuals / attribution)
      - HyperLogic head: DR-Net logits (2-class: Deny/Allow) + symbolic rule weights
    """

    def __init__(
        self,
        nr_features: int,
        nr_classes: int,
        n_rules: int,
        *,
        hyconex_nr_blocks: int = 4,
        hyconex_hidden: int = 256,
        hyconex_dropout: float = 0.25,
        hyperlogic_n_mc: int = 1,
        hyperlogic_tau: float = 0.1,
    ):
        super().__init__()
        if nr_classes != 2:
            raise ValueError(
                "DualExplainModel: HyperLogic DR head is binary (Deny/Allow). Use nr_classes=2."
            )
        self.nr_features = nr_features
        self.nr_classes = nr_classes
        self.n_rules = n_rules

        self.hyconex_net = HyperNet(
            nr_features=nr_features,
            nr_classes=nr_classes,
            nr_blocks=hyconex_nr_blocks,
            hidden_size=hyconex_hidden,
            dropout_rate=hyconex_dropout,
        )
        self.hyperlogic_branch = HyperLogicBranch(
            dim=nr_features,
            n_rules=n_rules,
            n_mc=hyperlogic_n_mc,
            tau=hyperlogic_tau,
        )

    def forward(
        self,
        x_hyconex: torch.Tensor,
        x_hyperlogic_pm: torch.Tensor,
        *,
        return_weights: bool = True,
    ) -> Dict[str, Any]:
        """
        x_hyconex: (B, D) floats in [0,1] (or scaled tabular)
        x_hyperlogic_pm: (B, D) in {-1, +1}
        """
        logits_hc, weights_hc = self.hyconex_net(
            x_hyconex, return_weights=return_weights, simple_weights=True
        )
        logits_hl, w_hl, u_hl = self.hyperlogic_branch(x_hyperlogic_pm)

        return {
            "logits_hyconex": logits_hc,
            "weights_hyconex": weights_hc,
            "logits_hyperlogic": logits_hl,
            "weights_rule": w_hl,
            "weights_or": u_hl,
        }
