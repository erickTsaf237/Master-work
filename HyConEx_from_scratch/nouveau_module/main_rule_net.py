from __future__ import annotations

import torch
import torch.nn.functional as F


def unpack_main_params(
    theta: torch.Tensor,
    input_dim: int,
    num_rules: int,
    num_classes: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """theta : [P] ou [B, P] → w_rule, b_rule, w_out, b_out (avec ou sans batch)."""
    batched = theta.dim() == 2
    if not batched:
        theta = theta.unsqueeze(0)

    bsz = theta.shape[0]
    idx = 0
    s = input_dim * num_rules
    w_rule = theta[:, idx : idx + s].view(bsz, input_dim, num_rules)
    idx += s

    b_rule = theta[:, idx : idx + num_rules]
    idx += num_rules

    s2 = num_rules * num_classes
    w_out = theta[:, idx : idx + s2].view(bsz, num_rules, num_classes)
    idx += s2

    b_out = theta[:, idx : idx + num_classes]

    if not batched:
        return w_rule.squeeze(0), b_rule.squeeze(0), w_out.squeeze(0), b_out.squeeze(0)
    return w_rule, b_rule, w_out, b_out


def main_logits_from_weights(
    x_bin: torch.Tensor,
    theta_main: torch.Tensor,
    input_dim: int,
    num_rules: int,
    num_classes: int,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
    """HyperLogic/DR-Net: x_bin ∈ {-1,+1}^D, u_k = x^T w_k - ||w_k||_1 + b_k, o_k = exp(-u_k^2/τ)."""
    w_rule, b_rule, w_out, b_out = unpack_main_params(theta_main, input_dim, num_rules, num_classes)
    batched_theta = w_rule.dim() == 3

    if batched_theta:
        abs_sum = w_rule.abs().sum(dim=1)
        u = torch.einsum("bd,bdk->bk", x_bin, w_rule) - abs_sum + b_rule
    else:
        abs_sum = w_rule.abs().sum(dim=0)
        u = x_bin @ w_rule - abs_sum.unsqueeze(0) + b_rule.unsqueeze(0)

    tau = max(float(temperature), 1e-6)
    rule_act = torch.exp(-(u * u) / tau)

    if batched_theta:
        logits = torch.einsum("bk,bkc->bc", rule_act, w_out) + b_out
    else:
        logits = rule_act @ w_out + b_out

    return logits, rule_act, (w_rule, b_rule, w_out, b_out)


def extract_rules(
    w_rule: torch.Tensor,
    w_out: torch.Tensor,
    binary_feature_names: list[str],
    class_names: list[str],
    top_per_rule: int = 4,
    min_abs_weight: float = 0.05,
) -> list[dict]:
    if w_rule.dim() == 3:
        w_rule = w_rule.mean(dim=0)
        w_out = w_out.mean(dim=0)

    rules: list[dict] = []
    w_rule_np = w_rule.detach().cpu().numpy()
    w_out_np = w_out.detach().cpu().numpy()

    for k in range(w_rule_np.shape[1]):
        wr = w_rule_np[:, k]
        active = [i for i in range(wr.shape[0]) if abs(wr[i]) >= min_abs_weight]
        if not active:
            continue
        active_sorted = sorted(active, key=lambda i: abs(wr[i]), reverse=True)[:top_per_rule]
        conds = []
        for i in active_sorted:
            literal = "+1" if wr[i] >= 0 else "-1"
            conds.append(f"{binary_feature_names[i]}={literal}")

        target_class_idx = int(w_out_np[k].argmax())
        confidence = float(F.softmax(torch.tensor(w_out_np[k]), dim=0)[target_class_idx].item())
        rules.append(
            {
                "rule_id": k,
                "if": conds,
                "then_class": class_names[target_class_idx],
                "score": confidence,
            }
        )

    rules.sort(key=lambda r: r["score"], reverse=True)
    return rules
