"""Entraînement RuleConEx sur jeux DLBAC one-hot (GPU par défaut)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from hyconex_from_scratch.trainer import set_seed
from ruleconex.config import RuleConExConfig
from ruleconex.loss import ruleconex_loss
from ruleconex.model import RuleConExModel
from ruleconex.utils import numpy_to_model_input


@dataclass
class TrainingResult:
    history: list[dict[str, float]] = field(default_factory=list)
    best_val_accuracy: float = 0.0
    best_state: dict[str, Any] | None = None
    class_names: list[str] = field(default_factory=list)
    feature_names: list[str] = field(default_factory=list)


def _auto_tune_config(cfg: RuleConExConfig, input_dim: int, num_classes: int) -> None:
    """Réglages mémoire GPU pour haute dimension (Amazon one-hot)."""
    if input_dim <= 512:
        return
    cfg.batch_size = min(cfg.batch_size, 32)
    cfg.eval_batch_size = min(cfg.eval_batch_size, 64)
    cfg.mc_train_samples = min(cfg.mc_train_samples, 2)
    cfg.mc_infer_samples = min(cfg.mc_infer_samples, 3)
    cfg.use_deep_branch = False
    cfg.hidden_dim = min(cfg.hidden_dim, 96)
    cfg.latent_dim = min(cfg.latent_dim, 48)
    cfg.num_rules = min(cfg.num_rules, 32)
    if num_classes == 2:
        cfg.epochs = min(cfg.epochs, 25)


class RuleConExTrainer:
    def __init__(self, config: RuleConExConfig | None = None, *, device: str | torch.device | None = None) -> None:
        self.config = config or RuleConExConfig()
        if device is None:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requis : aucun GPU détecté. Lancez avec un GPU disponible.")
            device = "cuda"
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError(f"RuleConEx doit s'exécuter sur GPU ; device={self.device} refusé.")
        self.model: RuleConExModel | None = None
        self.history: list[dict[str, float]] = []
        self.class_names: list[str] = []
        self.feature_names: list[str] = []

    def _build_model(self, input_dim: int, num_classes: int) -> RuleConExModel:
        cfg = self.config
        model = RuleConExModel(
            input_dim=input_dim,
            num_classes=num_classes,
            latent_dim=cfg.latent_dim,
            hidden_dim=cfg.hidden_dim,
            num_rules=cfg.num_rules,
            temperature=cfg.temperature,
            cf_hidden_dim=cfg.hidden_dim,
            tabresnet_blocks=cfg.tabresnet_blocks,
            dropout=cfg.dropout,
            use_deep_branch=cfg.use_deep_branch,
            hyconex_weight=cfg.hyconex_weight,
            rules_weight=cfg.rules_weight,
            deep_weight=cfg.deep_weight,
            mc_train_samples=cfg.mc_train_samples,
            mc_infer_samples=cfg.mc_infer_samples,
        ).to(self.device)
        self.model = model
        return model

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        *,
        feature_names: list[str] | None = None,
        class_names: list[str] | None = None,
        verbose: bool = True,
    ) -> TrainingResult:
        cfg = self.config
        set_seed(cfg.seed)

        X_train = numpy_to_model_input(X_train)
        y_train = np.asarray(y_train, dtype=np.int64)
        num_classes = int(np.max(y_train) + 1)
        input_dim = X_train.shape[1]
        _auto_tune_config(cfg, input_dim, num_classes)

        self.feature_names = feature_names or [f"oh_{i}" for i in range(input_dim)]
        self.class_names = class_names or [str(i) for i in range(num_classes)]

        if verbose:
            print(
                f"  GPU={torch.cuda.get_device_name(0)} | batch={cfg.batch_size} "
                f"| rules={cfg.num_rules} | MC={cfg.mc_train_samples}",
                flush=True,
            )

        model = self._build_model(input_dim, num_classes)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

        class_weights = None
        if cfg.use_class_weights and num_classes == 2:
            counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
            weights = counts.sum() / (num_classes * np.maximum(counts, 1.0))
            class_weights = torch.tensor(weights, dtype=torch.float32, device=self.device)

        train_ds = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long),
        )
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)

        has_val = X_val is not None and y_val is not None
        if has_val:
            X_val = numpy_to_model_input(X_val)
            y_val = np.asarray(y_val, dtype=np.int64)

        best_val_acc = -1.0
        best_state = None
        self.history = []

        for epoch in range(1, cfg.epochs + 1):
            model.train()
            running = 0.0
            n_batches = 0
            for xb, yb in train_loader:
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                pack = model.forward_pack(xb, mc_samples=cfg.mc_train_samples)
                breakdown = ruleconex_loss(model, pack, yb, xb, cfg, class_weights=class_weights)
                breakdown.total.backward()
                if cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()
                running += float(breakdown.total.item())
                n_batches += 1
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()

            train_loss = running / max(n_batches, 1)
            train_acc, train_ce = self._accuracy_loss(model, X_train, y_train)

            val_acc = float("nan")
            val_loss = float("nan")
            if has_val:
                val_acc, val_loss = self._accuracy_loss(model, X_val, y_val)

            row = {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "train_loss_ce": train_ce,
                "val_accuracy": val_acc,
                "val_loss": val_loss,
            }
            self.history.append(row)

            if verbose and (epoch == 1 or epoch % 5 == 0 or epoch == cfg.epochs):
                msg = f"epoch {epoch:3d} | loss={train_loss:.4f} train_acc={train_acc:.3f}"
                if has_val:
                    msg += f" val_acc={val_acc:.3f}"
                print(msg, flush=True)

            if has_val and val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if best_state is not None:
            model.load_state_dict(best_state)

        return TrainingResult(
            history=self.history,
            best_val_accuracy=best_val_acc,
            best_state=best_state,
            class_names=self.class_names,
            feature_names=self.feature_names,
        )

    def _accuracy_loss(
        self,
        model: RuleConExModel,
        X: np.ndarray,
        y: np.ndarray,
    ) -> tuple[float, float]:
        model.eval()
        y = np.asarray(y, dtype=np.int64)
        bs = self.config.eval_batch_size
        correct = 0
        total = 0
        ce_sum = 0.0

        with torch.no_grad():
            for start in range(0, len(X), bs):
                xb = torch.tensor(X[start : start + bs], dtype=torch.float32, device=self.device)
                yb = torch.tensor(y[start : start + bs], dtype=torch.long, device=self.device)
                logits = model(xb)
                ce_sum += float(F.cross_entropy(logits, yb, reduction="sum").item())
                correct += int((logits.argmax(dim=1) == yb).sum().item())
                total += int(yb.shape[0])

        return correct / max(total, 1), ce_sum / max(total, 1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self.model is not None
        self.model.eval()
        X = numpy_to_model_input(X)
        bs = self.config.eval_batch_size
        chunks: list[np.ndarray] = []

        with torch.no_grad():
            for start in range(0, len(X), bs):
                xb = torch.tensor(X[start : start + bs], dtype=torch.float32, device=self.device)
                logits = self.model(xb)
                chunks.append(torch.softmax(logits, dim=1).cpu().numpy())

        return np.concatenate(chunks, axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)
