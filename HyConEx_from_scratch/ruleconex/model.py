"""
RuleConEx : un hyperréseau unique génère tous les poids θ pour :
- branche HyConEx (importances locales + contrefactuels x' = x - W_c)
- branche HyperLogic (DR-Net / règles IF-THEN, échantillonnage Monte Carlo)
- branche deep optionnelle (TabResNet léger)

Haute dimension (Amazon one-hot > 512) : hyperréseau + règles sur l'espace latent z.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from nouveau_module.cf_head import generate_cf_binary
from nouveau_module.hypernet import TabResNetBackbone
from nouveau_module.main_rule_net import main_logits_from_weights


def to_bipolar(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    if x.min() >= -1.01 and x.max() <= 1.01:
        return torch.where(x > 0.25, torch.ones_like(x), -torch.ones_like(x))
    return torch.where(x > 0.5, torch.ones_like(x), -torch.ones_like(x))


def split_enc_bipolar(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x = x.float()
    if x.min() < -0.01:
        x_bin = to_bipolar(x)
        enc_in = (x_bin + 1.0) * 0.5
    else:
        enc_in = x.clamp(0.0, 1.0)
        x_bin = to_bipolar(enc_in)
    return enc_in, x_bin


def _main_theta_dim(input_dim: int, num_rules: int, num_classes: int) -> int:
    return input_dim * num_rules + num_rules + num_rules * num_classes + num_classes


@dataclass
class RuleConExForwardPack:
    logits: torch.Tensor
    logits_hyconex: torch.Tensor
    logits_rules: torch.Tensor
    logits_deep: torch.Tensor | None
    rule_activations: torch.Tensor
    z: torch.Tensor
    hyconex_weights: torch.Tensor
    hyconex_bias: torch.Tensor
    hyconex_contributions: torch.Tensor
    input_importance: torch.Tensor
    theta_main: torch.Tensor
    theta_cf: torch.Tensor
    rule_params: tuple[torch.Tensor, ...]
    mc_logits_rules: list[torch.Tensor] = field(default_factory=list)
    enc_in: torch.Tensor | None = None
    x_bin: torch.Tensor | None = None
    dr_input: torch.Tensor | None = None
    rules_on_input: bool = True


class RuleConExHyperNetwork(nn.Module):
    """Hyperréseau unique (TabResNet) → θ_hyconex, θ_rules, θ_cf par échantillon."""

    def __init__(
        self,
        hyper_input_dim: int,
        num_classes: int,
        latent_dim: int,
        num_rules: int,
        hidden_dim: int,
        cf_hidden_dim: int,
        *,
        n_blocks: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hyper_input_dim = hyper_input_dim
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.num_rules = num_rules
        self.cf_hidden_dim = cf_hidden_dim

        self.tab = TabResNetBackbone(hyper_input_dim, hidden_dim, n_blocks, dropout=dropout)

        hyconex_params = num_classes * (latent_dim + 1)
        main_params = _main_theta_dim(hyper_input_dim, num_rules, num_classes)
        cf_in = hyper_input_dim + num_classes
        cf_params = (cf_in * cf_hidden_dim) + cf_hidden_dim + (cf_hidden_dim * hyper_input_dim) + hyper_input_dim

        self.hyconex_head = nn.Linear(hidden_dim, hyconex_params)
        self.main_head = nn.Linear(hidden_dim, main_params)
        self.cf_head = nn.Linear(hidden_dim, cf_params)
        self.mc_noise_scale = nn.Parameter(torch.tensor(0.08))

    def forward(
        self,
        hyper_in: torch.Tensor,
        *,
        mc_samples: int = 1,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        h = self.tab(hyper_in)
        bsz = h.shape[0]

        hyconex_flat = self.hyconex_head(h)
        hyconex = hyconex_flat.view(bsz, self.num_classes, self.latent_dim + 1)
        theta_cf = self.cf_head(h)

        theta_list: list[torch.Tensor] = []
        noise_scale = self.mc_noise_scale.abs().clamp(0.01, 0.25)
        for m in range(mc_samples):
            h_m = h if m == 0 else h + torch.randn_like(h) * noise_scale
            theta_list.append(self.main_head(h_m))

        return {
            "h": h,
            "hyconex": hyconex,
            "theta_cf": theta_cf,
            "theta_main": theta_list[0],
            "theta_main_mc": theta_list,
        }


class RuleConExModel(nn.Module):
    """Classifieur interprétable grant/deny DLBAC (une passe forward)."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        *,
        latent_dim: int = 64,
        hidden_dim: int = 128,
        num_rules: int = 48,
        temperature: float = 0.7,
        cf_hidden_dim: int = 128,
        tabresnet_blocks: int = 3,
        dropout: float = 0.1,
        use_deep_branch: bool = True,
        hyconex_weight: float = 0.45,
        rules_weight: float = 0.45,
        deep_weight: float = 0.15,
        mc_train_samples: int = 3,
        mc_infer_samples: int = 5,
        cf_subtract_scale: float = 0.35,
        max_drnet_input_dim: int = 512,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.num_rules = num_rules
        self.temperature = temperature
        self.cf_hidden_dim = cf_hidden_dim
        self.use_deep_branch = use_deep_branch
        self.mc_train_samples = mc_train_samples
        self.mc_infer_samples = mc_infer_samples
        self.cf_subtract_scale = cf_subtract_scale
        self.max_drnet_input_dim = max_drnet_input_dim
        self.rules_on_input = input_dim <= max_drnet_input_dim
        self.hyper_input_dim = input_dim if self.rules_on_input else latent_dim

        self.enc_stem = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.enc_head = nn.Linear(hidden_dim, latent_dim)
        self.imp_rank = 32 if input_dim > max_drnet_input_dim else 0
        if self.imp_rank > 0:
            self.importance_up = nn.Linear(hidden_dim, num_classes * self.imp_rank)
            self.importance_down = nn.Linear(self.imp_rank, input_dim, bias=False)
        else:
            self.importance_proj = nn.Linear(hidden_dim, num_classes * input_dim)

        self.hypernet = RuleConExHyperNetwork(
            self.hyper_input_dim,
            num_classes,
            latent_dim,
            num_rules,
            hidden_dim,
            cf_hidden_dim,
            n_blocks=tabresnet_blocks,
            dropout=dropout,
        )

        if use_deep_branch:
            deep_in = input_dim if self.rules_on_input else latent_dim
            self.deep_stem = TabResNetBackbone(deep_in, hidden_dim, max(2, tabresnet_blocks - 1), dropout)
            self.deep_head = nn.Linear(hidden_dim, num_classes)
        else:
            self.deep_stem = None
            self.deep_head = None

        raw = torch.tensor([hyconex_weight, rules_weight, deep_weight if use_deep_branch else 0.0])
        self.branch_logits = nn.Parameter(raw.log().clone())

    def encode(self, enc_in: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.enc_stem(enc_in)
        z = self.enc_head(h)
        return h, z

    def branch_weights(self) -> tuple[float, float, float]:
        w = F.softmax(self.branch_logits, dim=0)
        if not self.use_deep_branch:
            s = w[0] + w[1]
            return float(w[0] / s), float(w[1] / s), 0.0
        return float(w[0]), float(w[1]), float(w[2])

    def _hyconex_logits(
        self,
        z: torch.Tensor,
        hyconex: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        w = hyconex[:, :, : self.latent_dim]
        b = hyconex[:, :, self.latent_dim]
        contributions = w * z.unsqueeze(1)
        logits = contributions.sum(dim=2) + b
        return logits, w, b, contributions

    def _dr_input(self, x_bin: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        if self.rules_on_input:
            return x_bin
        return to_bipolar(z)

    def forward_pack(
        self,
        x: torch.Tensor,
        *,
        mc_samples: int | None = None,
    ) -> RuleConExForwardPack:
        enc_in, x_bin = split_enc_bipolar(x)
        h_enc, z = self.encode(enc_in)

        if mc_samples is None:
            mc_samples = self.mc_train_samples if self.training else self.mc_infer_samples

        dr_in = self._dr_input(x_bin, z)
        hyper = self.hypernet(dr_in, mc_samples=mc_samples)

        logits_hyconex, w_h, b_h, contributions = self._hyconex_logits(z, hyper["hyconex"])  # type: ignore[arg-type]

        mc_logits: list[torch.Tensor] = []
        mc_acts: list[torch.Tensor] = []
        rule_params_last: tuple[torch.Tensor, ...] | None = None

        for theta_m in hyper["theta_main_mc"]:  # type: ignore[union-attr]
            logits_m, act_m, rp = main_logits_from_weights(
                dr_in,
                theta_m,
                input_dim=self.hyper_input_dim,
                num_rules=self.num_rules,
                num_classes=self.num_classes,
                temperature=self.temperature,
            )
            mc_logits.append(logits_m)
            mc_acts.append(act_m)
            rule_params_last = rp

        logits_rules = torch.stack(mc_logits, dim=0).mean(dim=0)
        rule_act = torch.stack(mc_acts, dim=0).mean(dim=0)

        logits_deep = None
        if self.use_deep_branch and self.deep_stem is not None and self.deep_head is not None:
            deep_in = enc_in if self.rules_on_input else z
            logits_deep = self.deep_head(self.deep_stem(deep_in))

        wh, wr, wd = self.branch_weights()
        logits = wh * logits_hyconex + wr * logits_rules
        if logits_deep is not None:
            logits = logits + wd * logits_deep

        if self.imp_rank > 0:
            imp_core = self.importance_up(h_enc).view(-1, self.num_classes, self.imp_rank)
            imp_feat = self.importance_down(imp_core)
            importance = imp_feat.abs() * enc_in.unsqueeze(1)
        else:
            imp_raw = self.importance_proj(h_enc).view(-1, self.num_classes, self.input_dim)
            importance = imp_raw.abs() * enc_in.unsqueeze(1)

        assert rule_params_last is not None
        return RuleConExForwardPack(
            logits=logits,
            logits_hyconex=logits_hyconex,
            logits_rules=logits_rules,
            logits_deep=logits_deep,
            rule_activations=rule_act,
            z=z,
            hyconex_weights=w_h,
            hyconex_bias=b_h,
            hyconex_contributions=contributions,
            input_importance=importance,
            theta_main=hyper["theta_main"],  # type: ignore[arg-type]
            theta_cf=hyper["theta_cf"],  # type: ignore[arg-type]
            rule_params=rule_params_last,
            mc_logits_rules=mc_logits,
            enc_in=enc_in,
            x_bin=x_bin,
            dr_input=dr_in,
            rules_on_input=self.rules_on_input,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_pack(x).logits

    def counterfactual_subtract(
        self,
        enc_in: torch.Tensor,
        y_target: torch.Tensor,
        importance: torch.Tensor,
    ) -> torch.Tensor:
        bsz = enc_in.shape[0]
        idx = torch.arange(bsz, device=enc_in.device)
        w_c = importance[idx, y_target]
        return torch.clamp(enc_in - self.cf_subtract_scale * w_c, 0.0, 1.0)

    def generate_counterfactual(
        self,
        x: torch.Tensor,
        y_target: torch.Tensor,
        *,
        pack: RuleConExForwardPack | None = None,
    ) -> torch.Tensor:
        if pack is None:
            pack = self.forward_pack(x)
        assert pack.enc_in is not None and pack.dr_input is not None

        x_cf_sub = self.counterfactual_subtract(pack.enc_in, y_target, pack.input_importance)

        x_cf_bin = generate_cf_binary(
            pack.dr_input,
            y_target,
            pack.theta_cf,
            self.hyper_input_dim,
            self.num_classes,
            self.cf_hidden_dim,
        )
        if self.rules_on_input:
            x_cf_flip = (x_cf_bin + 1.0) * 0.5
            return 0.55 * x_cf_sub + 0.45 * x_cf_flip
        return x_cf_sub

    def generate_counterfactuals_all_classes(
        self,
        x: torch.Tensor,
        y_current: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pack = self.forward_pack(x)
        bsz = x.shape[0]
        num_classes = self.num_classes

        x_cf_list = [
            self.generate_counterfactual(x, torch.full((bsz,), c, device=x.device, dtype=torch.long), pack=pack)
            for c in range(num_classes)
        ]
        x_cf = torch.stack(x_cf_list, dim=1)
        logits_cf = self.forward(x_cf.reshape(bsz * num_classes, -1)).view(bsz, num_classes, num_classes)
        return x_cf, logits_cf
