"""
Phased training: (1) HyConEx head only, (2) HyperLogic head only, (3) joint + consistency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .dual_model import DualExplainModel
from .eval_metrics import evaluate_model
from .losses import combined_loss, hyconex_counterfactual_ce
from .preprocessing import AccessControlPreprocessor


@dataclass
class TrainConfig:
    epochs_phase1: int = 40
    epochs_phase2: int = 40
    epochs_phase3: int = 60
    lr: float = 1e-3
    batch_size: int = 64
    lambda_consistency: float = 0.15
    lambda_cf: float = 0.05
    weight_decay: float = 1e-5
    device: str = "cpu"


def prepare_tensors(
    pre: AccessControlPreprocessor | None, X_np: np.ndarray, y_np: np.ndarray
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if pre is None:
        # Data already transformed (e.g., HyConEx AbstractDataset output).
        X_hc = torch.from_numpy(np.asarray(X_np, dtype=np.float32))
        X_pm = torch.where(X_hc >= 0.0, 1.0, -1.0).to(dtype=torch.float32)
    else:
        X_hc = pre.transform_hyconex(X_np)
        X_pm = pre.transform_hyperlogic(X_np)
    y = torch.from_numpy(np.asarray(y_np, dtype=np.int64)).long()
    return X_hc, X_pm, y


def _loader(
    X_hc: torch.Tensor, X_pm: torch.Tensor, y: torch.Tensor, batch_size: int, shuffle: bool
) -> DataLoader:
    return DataLoader(
        TensorDataset(X_hc, X_pm, y),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
    )


def train_phased(
    model: DualExplainModel,
    pre: AccessControlPreprocessor | None,
    X_train_np: np.ndarray,
    y_train_np: np.ndarray,
    X_val_np: np.ndarray,
    y_val_np: np.ndarray,
    cfg: TrainConfig,
    flow: Optional[torch.nn.Module] = None,
) -> List[dict]:
    device = torch.device(cfg.device)
    model = model.to(device)

    X_train_hc, X_train_pm, y_train = prepare_tensors(pre, X_train_np, y_train_np)
    X_val_hc, X_val_pm, y_val = prepare_tensors(pre, X_val_np, y_val_np)

    X_train_hc, X_train_pm, y_train = (
        X_train_hc.to(device),
        X_train_pm.to(device),
        y_train.to(device),
    )
    X_val_hc, X_val_pm, y_val = (
        X_val_hc.to(device),
        X_val_pm.to(device),
        y_val.to(device),
    )

    loader = _loader(X_train_hc, X_train_pm, y_train, cfg.batch_size, shuffle=True)
    history: List[dict] = []

    def run_phase(
        tag: str,
        n_epochs: int,
        *,
        train_hc: bool,
        train_hl: bool,
        lam_cons: float,
        lam_cf: float,
    ):
        for p in model.hyconex_net.parameters():
            p.requires_grad = train_hc
        for p in model.hyperlogic_branch.parameters():
            p.requires_grad = train_hl
        params = [p for p in model.parameters() if p.requires_grad]
        optim = torch.optim.AdamW(
            params, lr=cfg.lr, weight_decay=cfg.weight_decay
        )

        for ep in range(n_epochs):
            model.train()
            epoch_loss = 0.0
            n_batches = 0
            for xh, xpm, yb in loader:
                optim.zero_grad(set_to_none=True)
                out = model(xh, xpm, return_weights=True)
                if train_hc and not train_hl:
                    loss = F.cross_entropy(out["logits_hyconex"], yb)
                    if lam_cf > 0:
                        loss = loss + lam_cf * hyconex_counterfactual_ce(
                            model, xh, out, yb, use_distance=True
                        )
                elif train_hl and not train_hc:
                    loss = F.cross_entropy(out["logits_hyperlogic"], yb)
                else:
                    loss, _ = combined_loss(
                        model,
                        xh,
                        xpm,
                        out,
                        y=yb,
                        lambda_consistency=lam_cons,
                        lambda_cf=lam_cf,
                        lambda_flow=0.0,
                        flow=flow,
                    )
                loss.backward()
                optim.step()
                epoch_loss += float(loss.detach().item())
                n_batches += 1

            m_hc = evaluate_model(model, X_val_hc, X_val_pm, y_val, head="hyconex")
            m_hl = evaluate_model(
                model, X_val_hc, X_val_pm, y_val, head="hyperlogic"
            )
            history.append(
                {
                    "phase": tag,
                    "epoch": ep,
                    "train_loss": epoch_loss / max(n_batches, 1),
                    "val_acc_hyconex": m_hc["accuracy"],
                    "val_auroc_hyconex": m_hc["auroc"],
                    "val_acc_hyperlogic": m_hl["accuracy"],
                    "val_auroc_hyperlogic": m_hl["auroc"],
                }
            )

    run_phase(
        "phase1_hyconex",
        cfg.epochs_phase1,
        train_hc=True,
        train_hl=False,
        lam_cons=0.0,
        lam_cf=cfg.lambda_cf,
    )
    run_phase(
        "phase2_hyperlogic",
        cfg.epochs_phase2,
        train_hc=False,
        train_hl=True,
        lam_cons=0.0,
        lam_cf=0.0,
    )
    run_phase(
        "phase3_joint",
        cfg.epochs_phase3,
        train_hc=True,
        train_hl=True,
        lam_cons=cfg.lambda_consistency,
        lam_cf=cfg.lambda_cf,
    )

    return history
