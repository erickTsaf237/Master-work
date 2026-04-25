"""Boucle d'entraînement HyperRuleEx."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, TextIO

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from grok_hyperruleex.losses import total_loss
from grok_hyperruleex.model import HyperRuleEx


def _eval_bce_acc(
    model: torch.nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> tuple[float, float]:
    """Retourne (loss BCE moyenne, accuracy) sur tout le tenseur."""
    model.eval()
    n = X.size(0)
    if n == 0:
        return 0.0, 0.0
    loss_sum = 0.0
    correct = 0
    with torch.no_grad():
        for i in range(0, n, batch_size):
            xb = X[i : i + batch_size].to(device)
            yb = y[i : i + batch_size].to(device)
            logits = model(xb)["logits"]
            loss_sum += float(F.cross_entropy(logits, yb, reduction="sum").item())
            correct += int((logits.argmax(-1) == yb).sum().item())
    return loss_sum / n, correct / n


def _acc_bar(acc: float, width: int = 24, lo: float = 0.5, hi: float = 0.85) -> str:
    """Barre ASCII : lo -> vide, hi -> plein."""
    t = (acc - lo) / max(hi - lo, 1e-6)
    t = max(0.0, min(1.0, t))
    filled = int(round(t * width))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


@dataclass
class TrainConfig:
    epochs: int = 20
    batch_size: int = 128
    lr: float = 1e-3
    device: str = "cpu"
    lambda_sparse: float = 0.05
    lambda_div: float = 0.02
    lambda_stab: float = 0.05
    verbose: bool = True
    log_every: int = 1
    log_file: Optional[TextIO] = None


def train_model(
    model: HyperRuleEx,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: Optional[torch.Tensor] = None,
    y_val: Optional[torch.Tensor] = None,
    cfg: TrainConfig | None = None,
) -> List[Dict[str, float]]:
    cfg = cfg or TrainConfig()
    device = torch.device(cfg.device)
    log_out = cfg.log_file if cfg.log_file is not None else sys.stdout

    def _log(msg: str) -> None:
        if cfg.verbose:
            print(msg, file=log_out, flush=True)

    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=cfg.batch_size,
        shuffle=True,
    )
    history: List[Dict[str, float]] = []

    if cfg.verbose:
        _log(
            "Epoch | TrainLoss | BCE_tr | sparse |  div  | stab  | BCE_val | Acc_val | Acc_tr | bar_val"
        )
        _log(
            "------+-----------+--------+--------+-------+-------+---------+---------+--------+---------"
        )

    for epoch in range(cfg.epochs):
        model.train()
        total = 0.0
        sum_bce = 0.0
        sum_sp = 0.0
        sum_div = 0.0
        sum_stab = 0.0
        n_batches = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            losses = total_loss(
                model,
                {"x": xb, "y": yb},
                lambda_sparse=cfg.lambda_sparse,
                lambda_div=cfg.lambda_div,
                lambda_stab=cfg.lambda_stab,
            )
            losses["total"].backward()
            opt.step()
            total += float(losses["total"].detach().cpu().item())
            sum_bce += float(losses["bce"].detach().cpu().item())
            sum_sp += float(losses["sparse"].detach().cpu().item())
            sum_div += float(losses["div"].detach().cpu().item())
            sum_stab += float(losses["stab"].detach().cpu().item())
            n_batches += 1

        nb = max(n_batches, 1)
        train_loss_mean = total / nb
        mean_bce = sum_bce / nb
        mean_sp = sum_sp / nb
        mean_div = sum_div / nb
        mean_stab = sum_stab / nb

        row: Dict[str, float] = {
            "epoch": float(epoch + 1),
            "train_loss": train_loss_mean,
            "train_bce": mean_bce,
            "train_sparse": mean_sp,
            "train_div": mean_div,
            "train_stab": mean_stab,
        }

        if X_val is not None and y_val is not None:
            val_bce, val_acc = _eval_bce_acc(
                model, X_val, y_val, device, cfg.batch_size
            )
            row["val_bce"] = val_bce
            row["val_acc"] = val_acc
        else:
            val_bce, val_acc = float("nan"), float("nan")

        train_bce_full, train_acc_full = _eval_bce_acc(
            model, X_train, y_train, device, cfg.batch_size
        )
        row["train_acc"] = train_acc_full
        row["train_bce_full"] = train_bce_full

        history.append(row)

        if cfg.verbose and (epoch + 1) % max(cfg.log_every, 1) == 0:
            bar = _acc_bar(val_acc) if not (val_acc != val_acc) else "[" + "?" * 24 + "]"
            _log(
                f"{epoch + 1:5d} | {train_loss_mean:9.4f} | {mean_bce:6.4f} | {mean_sp:6.4f} | "
                f"{mean_div:5.3f} | {mean_stab:5.3f} | {val_bce:7.4f} | {val_acc:7.4f} | "
                f"{train_acc_full:6.4f} | {bar}"
            )

    if cfg.verbose:
        last = history[-1]
        _log("")
        tail = (
            f"Resume dernier epoch: train_loss={last['train_loss']:.4f} "
            f"train_bce(batch)={last['train_bce']:.4f} "
            f"train_acc={last['train_acc']:.4f}"
        )
        if "val_bce" in last:
            tail += (
                f" val_bce={last['val_bce']:.4f} val_acc={last['val_acc']:.4f} "
                f"{_acc_bar(last['val_acc'])}"
            )
        _log(tail)

    return history
