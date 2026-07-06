"""Pertes combinées RuleConEx : classification, ConEx, règles, diversité KL, L1."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from hyconex_from_scratch.trainer import sample_alternative_targets
from ruleconex.config import RuleConExConfig
from ruleconex.model import RuleConExForwardPack, RuleConExModel


@dataclass
class LossBreakdown:
    total: torch.Tensor
    ce: torch.Tensor
    ce_rules: torch.Tensor
    ce_cf: torch.Tensor
    conex_l1: torch.Tensor
    conex_l2: torch.Tensor
    rule_sparsity: torch.Tensor
    kl_diversity: torch.Tensor
    importance_l1: torch.Tensor


def kl_diversity_loss(mc_logits: list[torch.Tensor], temperature: float = 1.0) -> torch.Tensor:
    """Encourage des jeux de règles diversifiés via symétrique KL entre paires de logits MC."""
    if len(mc_logits) < 2:
        return torch.tensor(0.0, device=mc_logits[0].device)

    t = max(float(temperature), 1e-6)
    probs = [F.softmax(l / t, dim=1) for l in mc_logits]
    log_probs = [F.log_softmax(l / t, dim=1) for l in mc_logits]

    kl_sum = torch.tensor(0.0, device=mc_logits[0].device)
    pairs = 0
    for i in range(len(probs)):
        for j in range(i + 1, len(probs)):
            kl_ij = F.kl_div(log_probs[i], probs[j], reduction="batchmean")
            kl_ji = F.kl_div(log_probs[j], probs[i], reduction="batchmean")
            kl_sum = kl_sum + kl_ij + kl_ji
            pairs += 2
    return kl_sum / max(pairs, 1)


def ruleconex_loss(
    model: RuleConExModel,
    pack: RuleConExForwardPack,
    y: torch.Tensor,
    x: torch.Tensor,
    cfg: RuleConExConfig,
    *,
    class_weights: torch.Tensor | None = None,
) -> LossBreakdown:
    ce = F.cross_entropy(pack.logits, y, weight=class_weights)
    ce_rules = F.cross_entropy(pack.logits_rules, y, weight=class_weights)

    y_alt = sample_alternative_targets(y, model.num_classes)
    x_cf = model.generate_counterfactual(x, y_alt, pack=pack)
    logits_cf = model(x_cf)
    ce_cf = F.cross_entropy(logits_cf, y_alt, weight=class_weights)

    enc_in = pack.enc_in
    if enc_in is None:
        enc_in = x.clamp(0.0, 1.0) if x.min() >= 0 else (to_bipolar_safe(x) + 1) * 0.5
    delta = x_cf - enc_in
    conex_l1 = delta.abs().mean()
    conex_l2 = (delta**2).mean()

    w_rule, _, _, _ = pack.rule_params
    rule_sparsity = w_rule.abs().mean()

    kl_div = kl_diversity_loss(pack.mc_logits_rules, temperature=cfg.temperature)
    importance_l1 = pack.input_importance.mean()

    total = (
        ce
        + cfg.conex_lambda * ce_rules
        + cfg.cf_lambda * ce_cf
        + cfg.flip_lambda * conex_l2
        + cfg.conex_lambda * conex_l1
        + cfg.rule_sparsity_lambda * rule_sparsity
        + cfg.kl_diversity_lambda * kl_div
        + cfg.importance_l1_lambda * importance_l1
    )

    return LossBreakdown(
        total=total,
        ce=ce,
        ce_rules=ce_rules,
        ce_cf=ce_cf,
        conex_l1=conex_l1,
        conex_l2=conex_l2,
        rule_sparsity=rule_sparsity,
        kl_diversity=kl_div,
        importance_l1=importance_l1,
    )


def to_bipolar_safe(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x > 0.25, torch.ones_like(x), -torch.ones_like(x))
