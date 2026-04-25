"""Prétrain puis fine-tune avec ramp-up des lambdas L_conEx."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader, TensorDataset

from .config import NoiseHyConExConfig
from .dataset_adapter import TabularFeatureLayout
from .flow_maf import ConditionalMAF
from .losses import compute_finetune_loss, compute_pretrain_loss
from .model import NoiseHyConEx


@dataclass
class TrainLoopConfig:
    batch_size: int = 256
    lr: float = 2e-3
    device: str = "cpu"
    log_every_epoch: int = 1


TrainConfig = TrainLoopConfig


def train_noise_hyconex(
    model: NoiseHyConEx,
    flow: ConditionalMAF,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    layout: TabularFeatureLayout,
    cfg: NoiseHyConExConfig,
    loop: TrainLoopConfig | TrainConfig | None = None,
    x_target_train: Optional[torch.Tensor] = None,
) -> List[Dict[str, Any]]:
    """
    Phase 1 : prétrain (CE [+ cluster si x_target]).
    Phase 2 : fine-tune avec L_conEx (epochs locaux 0..finetune_epochs-1 pour ramp).
    """
    loop = loop or TrainLoopConfig()
    device = torch.device(loop.device)
    model = model.to(device)
    flow = flow.to(device)
    opt = torch.optim.AdamW(list(model.parameters()) + list(flow.parameters()), lr=loop.lr)
    loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=loop.batch_size,
        shuffle=True,
    )
    history: List[Dict[str, Any]] = []

    global_epoch = 0

    def run_epoch(epoch_local: int, phase: str) -> Dict[str, Any]:
        nonlocal global_epoch
        freeze_g = phase == "pretrain" and cfg.freeze_generator_during_pretrain
        for p in model.generator.parameters():
            p.requires_grad = not freeze_g
        model.train()
        flow.train()
        tot = 0.0
        n_batches = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            eps = torch.randn(xb.size(0), cfg.noise_dim, device=device, dtype=xb.dtype)
            opt.zero_grad()
            if phase == "pretrain":
                xt_batch = None
                if x_target_train is not None:
                    idx = torch.arange(xb.size(0), device=device)
                    xt_batch = x_target_train[idx]
                out = compute_pretrain_loss(
                    model, xb, yb, eps, layout, cfg, epoch_local, x_target=xt_batch
                )
            else:
                out = compute_finetune_loss(
                    model, xb, yb, eps, layout, flow, cfg, epoch_local
                )
            out["total"].backward()
            opt.step()
            tot += float(out["total"].detach().cpu().item())
            n_batches += 1
        row: Dict[str, Any] = {
            "phase": phase,
            "epoch_local": epoch_local,
            "global_epoch": global_epoch,
            "train_loss": tot / max(n_batches, 1),
        }
        model.eval()
        flow.eval()
        with torch.no_grad():
            vbatches = 0
            vacc = 0.0
            for i in range(0, X_val.size(0), loop.batch_size):
                xv = X_val[i : i + loop.batch_size].to(device)
                yv = y_val[i : i + loop.batch_size].to(device)
                pred = model(xv).argmax(-1)
                vacc += (pred == yv).float().mean().item()
                vbatches += 1
            row["val_acc"] = vacc / max(vbatches, 1)
        global_epoch += 1
        return row

    for e in range(cfg.pretrain_epochs):
        h = run_epoch(e, "pretrain")
        history.append(h)
        if (e + 1) % max(loop.log_every_epoch, 1) == 0:
            print(
                f"[pretrain] ep {e + 1}/{cfg.pretrain_epochs} "
                f"loss={h['train_loss']:.4f} val_acc={h['val_acc']:.4f}"
            )

    for e in range(cfg.finetune_epochs):
        h = run_epoch(e, "finetune")
        history.append(h)
        if (e + 1) % max(loop.log_every_epoch, 1) == 0:
            print(
                f"[finetune] ep {e + 1}/{cfg.finetune_epochs} "
                f"loss={h['train_loss']:.4f} val_acc={h['val_acc']:.4f}"
            )

    return history
