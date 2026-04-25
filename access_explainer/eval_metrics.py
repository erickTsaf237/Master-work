"""Classification metrics for binary/multi-class evaluation."""

from __future__ import annotations

import numpy as np
import torch


@torch.no_grad()
def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = torch.argmax(logits, dim=-1)
    return (pred == y).float().mean().item()


def binary_auroc(probs_class1: np.ndarray, y: np.ndarray) -> float:
    """probs_class1: P(y=1), y in {0,1}."""
    from sklearn.metrics import roc_auc_score

    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, probs_class1))


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    X_hc: torch.Tensor,
    X_pm: torch.Tensor,
    y: torch.Tensor,
    head: str = "hyconex",
) -> dict:
    model.eval()
    out = model(X_hc, X_pm, return_weights=True)
    if head == "hyconex":
        logits = out["logits_hyconex"]
    else:
        logits = out["logits_hyperlogic"]
    probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
    acc = accuracy_from_logits(logits, y)
    roc = binary_auroc(probs, y.cpu().numpy())
    return {"accuracy": acc, "auroc": roc, "head": head}
