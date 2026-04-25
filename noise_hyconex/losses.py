"""L_conEx (CE CF + proximité + flow) et prétrain, sans wandb."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from .config import NoiseHyConExConfig
from .counterfactual_ops import get_counterfact_with_mask
from .dataset_adapter import TabularFeatureLayout
from .flow_maf import ConditionalMAF
from .model import NoiseHyConEx


def _ce_cf(
    *,
    output_cf_logit: torch.Tensor,
    target_values: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    m = mask.reshape(-1)
    logits = output_cf_logit[m]
    targets = target_values[m]
    return F.cross_entropy(logits, targets.long())


def _cluster_pretrain(
    x_cf: torch.Tensor,
    x_exp: torch.Tensor,
    mask: torch.Tensor,
    x_target: torch.Tensor,
    layout: TabularFeatureLayout,
    shape: Tuple[int, int, int],
) -> torch.Tensor:
    b, d_out, d_in = shape
    num_idx = layout.numerical_features
    cat_idx = layout.categorical_features
    x_cf_b = x_cf.reshape(b, d_out, d_in - 1)
    x_t = x_target.to(x_cf.device)
    cluster = (
        torch.norm(x_cf_b[:, :, num_idx] - x_t[:, :, num_idx], dim=-1, p=2.0) * mask
    )
    if cat_idx:
        cluster = cluster + (
            torch.norm(x_cf_b[:, :, cat_idx] - x_t[:, :, cat_idx], dim=-1, p=2.0) / 10.0
        ) * mask
    cluster = cluster + (
        torch.norm(x_cf_b[:, :, num_idx] - x_exp[:, :, num_idx], dim=-1, p=2.0) * (~mask)
    )
    if cat_idx:
        cluster = cluster + (
            torch.norm(x_cf_b[:, :, cat_idx] - x_exp[:, :, cat_idx], dim=-1, p=2.0) / 10.0
        ) * (~mask)
    return cluster.reshape(b * d_out).mean()


def compute_pretrain_loss(
    model: NoiseHyConEx,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: torch.Tensor,
    layout: TabularFeatureLayout,
    cfg: NoiseHyConExConfig,
    epoch: int,
    x_target: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    logits, w = model(x, eps, return_weights=True, simple_weights=True)
    base = F.cross_entropy(logits, y.long())
    if (
        x_target is not None
        and cfg.cluster_lambda > 0
        and epoch > cfg.cluster_start_epoch
    ):
        x_cf, x_exp, mask, _, shape = get_counterfact_with_mask(
            x,
            logits,
            w,
            layout,
            x.device,
            use_projection=cfg.use_projection,
            cat_temperature=cfg.cat_softmax_temperature,
        )
        cl = _cluster_pretrain(x_cf, x_exp, mask, x_target, layout, shape)
        lam = min(1.0, (epoch - cfg.cluster_start_epoch) / 200.0) * cfg.cluster_lambda
        scale = max(1.0, x.size(1) / 50.0)
        base = base + (lam / scale) * cl
    return {"total": base, "base": base.detach()}


def compute_finetune_loss(
    model: NoiseHyConEx,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: torch.Tensor,
    layout: TabularFeatureLayout,
    flow: ConditionalMAF,
    cfg: NoiseHyConExConfig,
    epoch: int,
) -> Dict[str, Any]:
    logits, w = model(x, eps, return_weights=True, simple_weights=True)
    x_cf, x_exp, mask, target_values, shape = get_counterfact_with_mask(
        x,
        logits,
        w,
        layout,
        x.device,
        use_projection=cfg.use_projection,
        cat_temperature=cfg.cat_softmax_temperature,
    )
    b, d_out, d_in = shape
    feature_scale = max(1.0, x_exp.shape[2] / 50.0)

    validity = torch.tensor(0.0, device=x.device)
    plausibility = torch.tensor(0.0, device=x.device)
    classification_loss = torch.tensor(0.0, device=x.device)
    class_lambda = 0.0
    if epoch > cfg.class_start_epoch and cfg.class_lambda != 0:
        # Même ε par exemple d'origine (répliqué sur les C contre-factuels)
        eps_rep = eps.unsqueeze(1).expand(-1, d_out, -1).reshape(b * d_out, -1)
        output_cf_logit = model(x_cf, eps_rep)
        output_cf_logit = output_cf_logit.view(b, d_out, d_out)
        output_cf = torch.softmax(output_cf_logit, dim=-1)
        y_cf = torch.argmax(output_cf, dim=-1)
        validity = (
            torch.sum((y_cf.reshape(-1) == target_values).float() * mask.reshape(-1).float())
            / max(b * (d_out - 1), 1)
        )
        output_cf_logit_flat = output_cf_logit.reshape(b * d_out, d_out)
        classification_loss = _ce_cf(
            output_cf_logit=output_cf_logit_flat,
            target_values=target_values,
            mask=mask,
        )
        class_lambda = min(
            1.0,
            (epoch - cfg.class_start_epoch) / max(cfg.class_warm_up_epochs, 1),
        )

    distance_loss = torch.tensor(0.0, device=x.device)
    distance_lambda = 0.0
    if epoch > cfg.dist_start_epoch and cfg.dist_lambda != 0:
        num_idx = layout.numerical_features
        cat_idx = layout.categorical_features
        x_cf_b = x_cf.reshape(b, d_out, d_in - 1)
        d_num = torch.linalg.norm(
            x_cf_b[:, :, num_idx] - x_exp[:, :, num_idx],
            dim=-1,
        )
        if cat_idx:
            d_num = d_num + torch.linalg.norm(
                x_cf_b[:, :, cat_idx] - x_exp[:, :, cat_idx],
                dim=-1,
            ) / 10.0
        distance_loss = d_num.reshape(b * d_out).mean()
        distance_lambda = min(
            1.0,
            (epoch - cfg.dist_start_epoch) / max(cfg.dist_warm_up_epochs, 1),
        )

    flow_loss = torch.tensor(0.0, device=x.device)
    f_lambda = 0.0
    if epoch > cfg.flow_start_epoch and cfg.flow_lambda != 0:
        tv = target_values.reshape(-1, 1).float()
        log_p = flow(x_cf, context=tv)
        flow_loss = F.relu(cfg.log_prob_threshold - log_p) * mask.reshape(b * d_out).float()
        valid = ~torch.isnan(flow_loss) & ~torch.isinf(flow_loss)
        if valid.any():
            flow_loss = (flow_loss * valid.float()).sum() / valid.float().sum().clamp(min=1.0)
        else:
            flow_loss = torch.tensor(0.0, device=x.device)
        plausibility = torch.sum(
            (log_p >= cfg.log_prob_threshold).float() * mask.reshape(b * d_out).float()
        ) / max(b * (d_out - 1), 1)
        f_lambda = min(
            1.0,
            (epoch - cfg.flow_start_epoch) / max(cfg.flow_warm_up_epochs, 1),
        )

    class_lambda *= cfg.class_lambda
    distance_lambda *= cfg.dist_lambda / feature_scale
    f_lambda *= cfg.flow_lambda / feature_scale

    base = F.cross_entropy(logits, y.long())
    main_loss = (
        base
        + class_lambda * classification_loss
        + distance_lambda * distance_loss
        + f_lambda * flow_loss
    )
    return {
        "total": main_loss,
        "base": base.detach(),
        "classification_loss": classification_loss.detach(),
        "distance_loss": distance_loss.detach(),
        "flow_loss": (f_lambda * flow_loss).detach(),
        "validity": validity.detach(),
        "plausibility": plausibility.detach(),
    }
