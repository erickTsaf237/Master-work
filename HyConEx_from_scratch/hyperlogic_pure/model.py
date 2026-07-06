"""
DR-Net pur (100 % règles, pas de tête linéaire).

Deux régimes HyperLogic :
- basse dimension (<= max_instance_dim) : hyperréseau TabResNet par échantillon + CF binaire
- haute dimension (Amazon one-hot) : theta global modulé par contexte batch + greffe CF phase 2
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from nouveau_module.cf_head import generate_cf_binary
from nouveau_module.hypernet import HyperWeightGenerator
from nouveau_module.main_rule_net import main_logits_from_weights


def _main_theta_dim(input_dim: int, num_rules: int, num_classes: int) -> int:
    return input_dim * num_rules + num_rules + num_rules * num_classes + num_classes


def _cf_theta_dim(input_dim: int, num_classes: int, hidden_dim: int) -> int:
    cf_in = input_dim + num_classes
    return (cf_in * hidden_dim) + hidden_dim + (hidden_dim * input_dim) + input_dim


def continuous_to_bipolar(x: torch.Tensor) -> torch.Tensor:
    """One-hot [0,1] ou déjà bipolar -> {-1,+1}."""
    if x.min() >= -1.01 and x.max() <= 1.01:
        return torch.where(x > 0.25, torch.ones_like(x), -torch.ones_like(x))
    return torch.where(x > 0.5, torch.ones_like(x), -torch.ones_like(x))


def bipolar_to_continuous(x_bin: torch.Tensor) -> torch.Tensor:
    return (x_bin + 1.0) * 0.5


@dataclass
class PureForwardPack:
    logits: torch.Tensor
    rule_activations: torch.Tensor
    theta_main: torch.Tensor
    theta_cf: torch.Tensor | None = None


class InstanceDRNetCore(nn.Module):
    """Fidèle HyperLogic : TabResNet -> theta_main/theta_cf par échantillon."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        *,
        num_rules: int,
        cf_hidden_dim: int,
        hyper_hidden_dim: int,
        temperature: float,
        n_blocks: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_dim_bin = input_dim
        self.num_classes = num_classes
        self.num_rules = num_rules
        self.cf_hidden_dim = cf_hidden_dim
        self.temperature = temperature
        self.hyper = HyperWeightGenerator(
            input_dim_bin=input_dim,
            num_classes=num_classes,
            num_rules=num_rules,
            cf_hidden_dim=cf_hidden_dim,
            hidden_dim=hyper_hidden_dim,
            n_blocks=n_blocks,
            dropout=dropout,
        )

    def forward(self, x_bin: torch.Tensor) -> PureForwardPack:
        theta_main, theta_cf = self.hyper(x_bin)
        logits, rule_act, _ = main_logits_from_weights(
            x_bin,
            theta_main,
            input_dim=self.input_dim_bin,
            num_rules=self.num_rules,
            num_classes=self.num_classes,
            temperature=self.temperature,
        )
        return PureForwardPack(logits=logits, rule_activations=rule_act, theta_main=theta_main, theta_cf=theta_cf)

    def predict_logits(self, x_bin: torch.Tensor) -> torch.Tensor:
        return self.forward(x_bin).logits

    def _cf_hyperlogic(self, x_bin: torch.Tensor, y_target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        theta_main, theta_cf = self.hyper(x_bin)
        x_cf = generate_cf_binary(
            x_bin,
            y_target,
            theta_cf=theta_cf,
            input_dim=self.input_dim_bin,
            num_classes=self.num_classes,
            hidden_dim=self.cf_hidden_dim,
        )
        logits_cf, _, _ = main_logits_from_weights(
            x_cf,
            theta_main,
            input_dim=self.input_dim_bin,
            num_rules=self.num_rules,
            num_classes=self.num_classes,
            temperature=self.temperature,
        )
        return x_cf, logits_cf

    def generate_counterfactual(self, x_bin: torch.Tensor, y_target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self._cf_hyperlogic(x_bin, y_target)

    def generate_counterfactuals_all_classes(
        self, x_bin: torch.Tensor, y_current: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz = x_bin.shape[0]
        device = x_bin.device
        x_list: list[torch.Tensor] = []
        log_list: list[torch.Tensor] = []
        for cls in range(self.num_classes):
            y_tgt = torch.full((bsz,), cls, dtype=torch.long, device=device)
            x_cf, logits_cf = self._cf_hyperlogic(x_bin, y_tgt)
            x_list.append(x_cf)
            log_list.append(logits_cf)
        return torch.stack(x_list, dim=1), torch.stack(log_list, dim=1)


class EmbedDRNetCore(nn.Module):
    """
    DR-Net pur sur espace encodé (Amazon haute dimension).
    Encodeur one-hot -> embed ; règles 100 % sur emb_* (pas de tête linéaire).
    Phase 2 : CF binaire HyperLogic sur l'embedding (128 dims, faisable en mémoire).
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        *,
        num_rules: int,
        embed_dim: int,
        cf_hidden_dim: int,
        temperature: float,
        ctx_hidden_dim: int,
        ctx_modulation: float,
    ) -> None:
        super().__init__()
        self.input_dim_raw = input_dim
        self.input_dim_bin = embed_dim
        self.num_classes = num_classes
        self.num_rules = num_rules
        self.cf_hidden_dim = cf_hidden_dim
        self.temperature = temperature
        self.ctx_modulation = ctx_modulation
        self.embed_dim = embed_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

        self.main_p = _main_theta_dim(embed_dim, num_rules, num_classes)
        self.cf_p = _cf_theta_dim(embed_dim, num_classes, cf_hidden_dim)
        self.theta_main_bias = nn.Parameter(torch.zeros(self.main_p))
        self.ctx_main = nn.Sequential(
            nn.Linear(embed_dim * 2, ctx_hidden_dim),
            nn.GELU(),
            nn.Linear(ctx_hidden_dim, self.main_p),
        )
        self.theta_cf_bias: nn.Parameter | None = None

    def init_cf_modules(self) -> None:
        if self.theta_cf_bias is not None:
            return
        self.theta_cf_bias = nn.Parameter(torch.zeros(self.cf_p))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x.float())

    def _theta_main_for_batch(self, emb_bin: torch.Tensor) -> torch.Tensor:
        if self.training and emb_bin.shape[0] <= 256:
            ctx = torch.cat([emb_bin.mean(dim=0), emb_bin.std(dim=0).clamp(min=1e-6)], dim=0)
            delta = self.ctx_main(ctx)
            return self.theta_main_bias + self.ctx_modulation * delta
        return self.theta_main_bias

    def forward(self, x: torch.Tensor) -> PureForwardPack:
        emb = self.encode(x)
        emb_bin = continuous_to_bipolar(emb)
        theta_main = self._theta_main_for_batch(emb_bin)
        logits, rule_act, _ = main_logits_from_weights(
            emb_bin,
            theta_main,
            input_dim=self.embed_dim,
            num_rules=self.num_rules,
            num_classes=self.num_classes,
            temperature=self.temperature,
        )
        pack_theta = (
            theta_main.unsqueeze(0).expand(emb_bin.shape[0], -1)
            if theta_main.dim() == 1
            else theta_main
        )
        return PureForwardPack(logits=logits, rule_activations=rule_act, theta_main=pack_theta, theta_cf=None)

    def predict_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x).logits

    def generate_counterfactual(
        self, x: torch.Tensor, y_target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.theta_cf_bias is None:
            raise RuntimeError("CF non initialisé — phase 2 requise (init_cf_modules)")
        emb = self.encode(x)
        emb_bin = continuous_to_bipolar(emb)
        theta_main = self._theta_main_for_batch(emb_bin)
        bsz = emb_bin.shape[0]
        theta_cf = self.theta_cf_bias.unsqueeze(0).expand(bsz, -1)
        emb_cf = generate_cf_binary(
            emb_bin,
            y_target,
            theta_cf=theta_cf,
            input_dim=self.embed_dim,
            num_classes=self.num_classes,
            hidden_dim=self.cf_hidden_dim,
        )
        logits_cf, _, _ = main_logits_from_weights(
            emb_cf,
            theta_main,
            input_dim=self.embed_dim,
            num_rules=self.num_rules,
            num_classes=self.num_classes,
            temperature=self.temperature,
        )
        return emb_cf, logits_cf

    def generate_counterfactuals_all_classes(
        self, x: torch.Tensor, y_current: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz = x.shape[0]
        device = x.device
        x_list: list[torch.Tensor] = []
        log_list: list[torch.Tensor] = []
        for cls in range(self.num_classes):
            y_tgt = torch.full((bsz,), cls, dtype=torch.long, device=device)
            emb_cf, logits_cf = self.generate_counterfactual(x, y_tgt)
            x_list.append(emb_cf)
            log_list.append(logits_cf)
        return torch.stack(x_list, dim=1), torch.stack(log_list, dim=1)


class GlobalDRNetCore(nn.Module):
    """
    DR-Net pur sur entrée haute dimension.
    Theta appris globalement + légère modulation par contexte batch (moyenne, écart-type).
    Phase 2 : greffe CF HyConEx (modules CF créés à la demande pour économiser la VRAM).
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        *,
        num_rules: int,
        cf_hidden_dim: int,
        temperature: float,
        ctx_hidden_dim: int,
        ctx_modulation: float,
        embed_dim: int,
        cf_graft_scale: float,
    ) -> None:
        super().__init__()
        self.input_dim_bin = input_dim
        self.num_classes = num_classes
        self.num_rules = num_rules
        self.cf_hidden_dim = cf_hidden_dim
        self.temperature = temperature
        self.ctx_modulation = ctx_modulation
        self.cf_graft_scale = cf_graft_scale
        self.embed_dim = embed_dim

        self.main_p = _main_theta_dim(input_dim, num_rules, num_classes)
        self.cf_p = _cf_theta_dim(input_dim, num_classes, cf_hidden_dim)

        self.theta_main_bias = nn.Parameter(torch.zeros(self.main_p))
        self.ctx_main = nn.Sequential(
            nn.Linear(input_dim * 2, ctx_hidden_dim),
            nn.GELU(),
            nn.Linear(ctx_hidden_dim, self.main_p),
        )

        self.theta_cf_bias: nn.Parameter | None = None
        self.encoder: nn.Sequential | None = None
        self.cf_graft: nn.Sequential | None = None

    def init_cf_modules(self) -> None:
        """Greffe CF (phase 2) — évite d'allouer ~4M params en phase 1."""
        if self.cf_graft is not None:
            return
        self.theta_cf_bias = nn.Parameter(torch.zeros(self.cf_p))
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim_bin, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.GELU(),
        )
        self.cf_graft = nn.Sequential(
            nn.Linear(self.embed_dim + self.num_classes, 128),
            nn.GELU(),
            nn.Linear(128, self.input_dim_bin),
            nn.Tanh(),
        )

    def _theta_main_for_batch(self, x_bin: torch.Tensor) -> torch.Tensor:
        """Entraînement : modulation légère par stats batch. Inférence : theta global fixe."""
        if self.training and x_bin.shape[0] <= 256:
            ctx = torch.cat([x_bin.mean(dim=0), x_bin.std(dim=0).clamp(min=1e-6)], dim=0)
            delta = self.ctx_main(ctx)
            return self.theta_main_bias + self.ctx_modulation * delta
        return self.theta_main_bias

    def _batch_theta_cf(self, x_bin: torch.Tensor) -> torch.Tensor:
        if self.theta_cf_bias is None:
            raise RuntimeError("Modules CF non initialisés — appelez init_cf_modules()")
        bsz = x_bin.shape[0]
        return self.theta_cf_bias.unsqueeze(0).expand(bsz, -1)

    def forward(self, x_bin: torch.Tensor) -> PureForwardPack:
        theta_main = self._theta_main_for_batch(x_bin)
        logits, rule_act, _ = main_logits_from_weights(
            x_bin,
            theta_main,
            input_dim=self.input_dim_bin,
            num_rules=self.num_rules,
            num_classes=self.num_classes,
            temperature=self.temperature,
        )
        pack_theta = (
            theta_main.unsqueeze(0).expand(x_bin.shape[0], -1)
            if theta_main.dim() == 1
            else theta_main
        )
        return PureForwardPack(logits=logits, rule_activations=rule_act, theta_main=pack_theta, theta_cf=None)

    def predict_logits(self, x_bin: torch.Tensor) -> torch.Tensor:
        return self.forward(x_bin).logits

    def generate_counterfactual_hyperlogic(
        self, x_bin: torch.Tensor, y_target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        theta_main = self._theta_main_for_batch(x_bin)
        theta_cf = self._batch_theta_cf(x_bin)
        x_cf = generate_cf_binary(
            x_bin,
            y_target,
            theta_cf=theta_cf,
            input_dim=self.input_dim_bin,
            num_classes=self.num_classes,
            hidden_dim=self.cf_hidden_dim,
        )
        logits_cf, _, _ = main_logits_from_weights(
            x_cf,
            theta_main,
            input_dim=self.input_dim_bin,
            num_rules=self.num_rules,
            num_classes=self.num_classes,
            temperature=self.temperature,
        )
        return x_cf, logits_cf

    def generate_counterfactual_graft(
        self, x_bin: torch.Tensor, y_target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.cf_graft is None or self.encoder is None:
            raise RuntimeError("Greffe CF non initialisée — appelez init_cf_modules()")
        x_01 = bipolar_to_continuous(x_bin)
        ctx = self.encoder(x_01)
        y_oh = F.one_hot(y_target, num_classes=self.num_classes).float()
        delta = self.cf_graft(torch.cat([ctx, y_oh], dim=1))
        x_cf_01 = torch.clamp(x_01 + self.cf_graft_scale * delta, 0.0, 1.0)
        x_cf_bin = continuous_to_bipolar(x_cf_01)
        theta_main = self._theta_main_for_batch(x_bin)
        logits_cf, _, _ = main_logits_from_weights(
            x_cf_bin,
            theta_main,
            input_dim=self.input_dim_bin,
            num_rules=self.num_rules,
            num_classes=self.num_classes,
            temperature=self.temperature,
        )
        return x_cf_bin, logits_cf

    def generate_counterfactual(
        self, x_bin: torch.Tensor, y_target: torch.Tensor, *, use_graft: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if use_graft:
            return self.generate_counterfactual_graft(x_bin, y_target)
        return self.generate_counterfactual_hyperlogic(x_bin, y_target)

    def generate_counterfactuals_all_classes(
        self, x_bin: torch.Tensor, y_current: torch.Tensor, *, use_graft: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz = x_bin.shape[0]
        device = x_bin.device
        x_list: list[torch.Tensor] = []
        log_list: list[torch.Tensor] = []
        for cls in range(self.num_classes):
            y_tgt = torch.full((bsz,), cls, dtype=torch.long, device=device)
            x_cf, logits_cf = self.generate_counterfactual(x_bin, y_tgt, use_graft=use_graft)
            x_list.append(x_cf)
            log_list.append(logits_cf)
        return torch.stack(x_list, dim=1), torch.stack(log_list, dim=1)


class PureDRNetModel(nn.Module):
    """Façade : DR-Net pur, sans tête linéaire."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        *,
        num_rules: int = 48,
        cf_hidden_dim: int = 128,
        hyper_hidden_dim: int = 128,
        temperature: float = 0.8,
        tabresnet_n_blocks: int = 4,
        tabresnet_dropout: float = 0.1,
        max_instance_dim: int = 512,
        ctx_hidden_dim: int = 256,
        ctx_modulation: float = 0.1,
        embed_dim_high: int = 128,
        cf_graft_scale: float = 0.35,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.num_rules = num_rules
        self.mode = "instance" if input_dim <= max_instance_dim else "embed"
        self.use_cf_graft = False

        if self.mode == "instance":
            self.core = InstanceDRNetCore(
                input_dim,
                num_classes,
                num_rules=num_rules,
                cf_hidden_dim=cf_hidden_dim,
                hyper_hidden_dim=hyper_hidden_dim,
                temperature=temperature,
                n_blocks=tabresnet_n_blocks,
                dropout=tabresnet_dropout,
            )
        else:
            self.core = EmbedDRNetCore(
                input_dim,
                num_classes,
                num_rules=num_rules,
                embed_dim=embed_dim_high,
                cf_hidden_dim=cf_hidden_dim,
                temperature=temperature,
                ctx_hidden_dim=ctx_hidden_dim,
                ctx_modulation=ctx_modulation,
            )

    @property
    def input_dim_bin(self) -> int:
        return self.core.input_dim_bin

    def forward(self, x: torch.Tensor) -> PureForwardPack:
        if self.mode == "instance":
            x_bin = continuous_to_bipolar(x)
            return self.core(x_bin)
        return self.core(x)

    def predict_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x).logits

    def generate_counterfactual(self, x: torch.Tensor, y_target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.mode == "instance":
            x_bin = continuous_to_bipolar(x)
            return self.core.generate_counterfactual(x_bin, y_target)
        return self.core.generate_counterfactual(x, y_target)

    def generate_counterfactuals_all_classes(
        self, x: torch.Tensor, y_current: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.mode == "instance":
            x_bin = continuous_to_bipolar(x)
            return self.core.generate_counterfactuals_all_classes(x_bin, y_current)
        return self.core.generate_counterfactuals_all_classes(x, y_current)
