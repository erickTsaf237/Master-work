from __future__ import annotations

import torch
import torch.nn.functional as F


def unpack_main_params(theta: torch.Tensor, input_dim: int, num_rules: int, num_classes: int) -> tuple[torch.Tensor, ...]:
    idx = 0
    s = input_dim * num_rules
    w_rule = theta[idx : idx + s].view(input_dim, num_rules)
    idx += s

    b_rule = theta[idx : idx + num_rules]
    idx += num_rules

    s2 = num_rules * num_classes
    w_out = theta[idx : idx + s2].view(num_rules, num_classes)
    idx += s2

    b_out = theta[idx : idx + num_classes]
    return w_rule, b_rule, w_out, b_out


def main_logits_from_weights(
    x_bin: torch.Tensor,
    theta_main: torch.Tensor,
    input_dim: int,
    num_rules: int,
    num_classes: int,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
    w_rule, b_rule, w_out, b_out = unpack_main_params(theta_main, input_dim, num_rules, num_classes)

    rule_pre = x_bin @ w_rule + b_rule
    rule_act = torch.sigmoid(rule_pre / max(temperature, 1e-6))
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
            sign = "ON" if wr[i] >= 0 else "OFF"
            conds.append(f"{binary_feature_names[i]}={sign}")

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
