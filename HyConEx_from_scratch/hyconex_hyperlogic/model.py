"""
Hybride HyConEx + HyperLogic pour DLBAC.

- Tete lineaire (signal fort sur one-hot sparse, comme SVM)
- Encodeur bottleneck + DR-Net a poids globaux (HyperLogic, pas hypernet par echantillon)
- Generateur de contrefactuels continu (HyConEx) sur features [0,1]
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from nouveau_module.main_rule_net import main_logits_from_weights


@dataclass
class ForwardPack:
    logits: torch.Tensor
    logits_linear: torch.Tensor
    logits_rules: torch.Tensor
    rule_activations: torch.Tensor
    embedding: torch.Tensor


class HyConExHyperLogicModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        *,
        embed_dim: int = 256,
        num_rules: int = 32,
        cf_hidden_dim: int = 128,
        temperature: float = 0.5,
        linear_weight: float = 0.65,
        rule_weight: float = 0.35,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.num_rules = num_rules
        self.cf_hidden_dim = cf_hidden_dim
        self.temperature = temperature
        self.linear_weight = linear_weight
        self.rule_weight = rule_weight

        # --- HyConEx : classifieur lineaire + encodeur ---
        self.linear_head = nn.Linear(input_dim, num_classes)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

        # --- HyperLogic DR-Net : poids globaux (conditionnes par contexte batch) ---
        self.ctx_to_theta = nn.Linear(embed_dim, self._theta_dim())
        self.theta_bias = nn.Parameter(torch.zeros(self._theta_dim()))

        # --- HyConEx CF : delta continu sur [0,1] ---
        self.cf_head = nn.Sequential(
            nn.Linear(embed_dim + num_classes, cf_hidden_dim),
            nn.GELU(),
            nn.Linear(cf_hidden_dim, cf_hidden_dim),
            nn.GELU(),
            nn.Linear(cf_hidden_dim, input_dim),
            nn.Tanh(),
        )

        self._init_weights()

    def _theta_dim(self) -> int:
        """DR-Net sur l'espace encode (embed_dim), pas sur les 14k colonnes one-hot."""
        d, k, c = self.embed_dim, self.num_rules, self.num_classes
        return d * k + k + k * c + c

    def _init_weights(self) -> None:
        nn.init.zeros_(self.linear_head.bias)
        nn.init.normal_(self.linear_head.weight, std=0.01)

    @staticmethod
    def to_bipolar(x: torch.Tensor) -> torch.Tensor:
        """[0,1] ou deja bipolar -> {-1,+1}."""
        if x.min() >= -1.01 and x.max() <= 1.01:
            return torch.where(x > 0.25, torch.ones_like(x), -torch.ones_like(x))
        return torch.where(x > 0.5, torch.ones_like(x), -torch.ones_like(x))

    def _context_embedding(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        return h

    def _theta_from_context(self, ctx: torch.Tensor) -> torch.Tensor:
        """Poids DR-Net : biais global + modulation legere par contexte (moyenne batch)."""
        batch_ctx = ctx.mean(dim=0, keepdim=True)
        delta = self.ctx_to_theta(batch_ctx)
        return self.theta_bias.unsqueeze(0) + 0.1 * delta

    def forward(self, x: torch.Tensor) -> ForwardPack:
        x = x.float()
        logits_linear = self.linear_head(x)
        ctx = self._context_embedding(x)
        ctx_bin = self.to_bipolar(ctx)
        theta = self._theta_from_context(ctx)
        logits_rules, rule_act, _ = main_logits_from_weights(
            ctx_bin,
            theta.squeeze(0),
            input_dim=self.embed_dim,
            num_rules=self.num_rules,
            num_classes=self.num_classes,
            temperature=self.temperature,
        )
        logits = self.linear_weight * logits_linear + self.rule_weight * logits_rules
        return ForwardPack(
            logits=logits,
            logits_linear=logits_linear,
            logits_rules=logits_rules,
            rule_activations=rule_act,
            embedding=ctx,
        )

    def predict_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x).logits

    def generate_counterfactual(self, x: torch.Tensor, y_target: torch.Tensor) -> torch.Tensor:
        """Contrefactuel continu HyConEx : x' = clamp(x + delta, 0, 1)."""
        ctx = self._context_embedding(x)
        y_oh = F.one_hot(y_target, num_classes=self.num_classes).float()
        delta = self.cf_head(torch.cat([ctx, y_oh], dim=1))
        scale = 0.35
        return torch.clamp(x + scale * delta, 0.0, 1.0)

    def generate_counterfactuals_all_classes(
        self, x: torch.Tensor, y_current: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pour chaque classe c != y, genere x_cf et logits(x_cf)."""
        num_classes = self.num_classes
        bsz = x.shape[0]
        x_rep = x.unsqueeze(1).expand(-1, num_classes, -1).reshape(bsz * num_classes, -1)
        class_ids = torch.arange(num_classes, device=x.device).view(1, num_classes).expand(bsz, -1)
        class_ids = class_ids.reshape(-1)
        x_cf = self.generate_counterfactual(x_rep, class_ids)
        logits_cf = self.predict_logits(x_cf).view(bsz, num_classes, num_classes)
        return x_cf.view(bsz, num_classes, -1), logits_cf
