from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from hyconex_from_scratch.config import TrainConfig
from hyconex_from_scratch.model import HyConExFromScratch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sample_alternative_targets(y: torch.Tensor, num_classes: int) -> torch.Tensor:
    noise = torch.randint(low=1, high=num_classes, size=y.shape, device=y.device)
    return (y + noise) % num_classes


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


@dataclass
class TrainingResult:
    """Résultat renvoyé par ``train()`` ou ``HyConExTrainer.fit()``."""

    model: HyConExFromScratch
    num_classes: int
    device: torch.device
    history: list[dict[str, float | int]]
    best_val_accuracy: float
    test_metrics: dict[str, Any] | None = None
    counterfactual_metrics: dict[str, Any] | None = None


class HyConExTrainer:
    """
    Entraîne ``HyConExFromScratch`` sur des tenseurs ou tableaux numpy ``X, y``.

    Les entrées doivent être en float32 pour ``X`` et entiers pour ``y`` (classes 0..C-1).
    Pour la génération de contre-factuels avec ``clamp``, il est recommandé de normaliser
    les features dans ``[0, 1]`` (ex. ``MinMaxScaler``).
    """

    def __init__(
        self,
        config: TrainConfig | None = None,
        *,
        device: str | torch.device | None = "auto",
    ) -> None:
        self.config = config or TrainConfig()
        self.device = _resolve_device(device)
        self.model: HyConExFromScratch | None = None
        self._num_classes: int | None = None
        self.history: list[dict[str, float | int]] = []

    def _ensure_model(self, input_dim: int, num_classes: int) -> HyConExFromScratch:
        cfg = self.config
        self._num_classes = num_classes
        self.model = HyConExFromScratch(
            input_dim=input_dim,
            num_classes=num_classes,
            latent_dim=cfg.latent_dim,
            hidden_dim=cfg.hidden_dim,
        ).to(self.device)
        return self.model

    def fit(
        self,
        X_train: np.ndarray | torch.Tensor,
        y_train: np.ndarray | torch.Tensor,
        X_val: np.ndarray | torch.Tensor | None = None,
        y_val: np.ndarray | torch.Tensor | None = None,
        *,
        verbose: bool = True,
    ) -> TrainingResult:
        cfg = self.config
        set_seed(cfg.seed)

        X_train = np.asarray(X_train, dtype=np.float32)
        y_train = np.asarray(y_train, dtype=np.int64)
        num_classes = int(np.max(y_train) + 1)

        model = self._ensure_model(X_train.shape[1], num_classes)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

        train_ds = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long),
        )
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)

        has_val = X_val is not None and y_val is not None
        if has_val:
            X_val = np.asarray(X_val, dtype=np.float32)
            y_val = np.asarray(y_val, dtype=np.int64)

        best_val_acc = -1.0
        best_state: dict[str, torch.Tensor] | None = None
        self.history = []

        for epoch in range(1, cfg.epochs + 1):
            model.train()
            running_loss = 0.0
            for xb, yb in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                y_target = sample_alternative_targets(yb, num_classes)

                logits = model(xb)
                ce = F.cross_entropy(logits, yb)

                x_cf = model.generate_counterfactual(xb, y_target)
                logits_cf = model(x_cf)
                ce_cf = F.cross_entropy(logits_cf, y_target)
                delta = x_cf - xb
                l1 = delta.abs().mean()
                l2 = (delta**2).mean()

                loss = ce + cfg.cf_lambda * ce_cf + cfg.l1_lambda * l1 + cfg.l2_lambda * l2

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * xb.shape[0]

            avg_loss = running_loss / len(train_ds)

            if has_val:
                val_metrics = self._evaluate_arrays(model, X_val, y_val)  # type: ignore[arg-type]
                val_acc = float(val_metrics["accuracy"])
            else:
                val_acc = float("nan")

            if has_val and val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

            self.history.append(
                {
                    "epoch": epoch,
                    "train_loss": float(avg_loss),
                    "val_accuracy": float(val_acc) if has_val else float("nan"),
                    "best_val_accuracy": float(best_val_acc) if has_val else float("nan"),
                }
            )

            if verbose:
                if has_val:
                    print(
                        f"[Epoch {epoch:03d}/{cfg.epochs}] "
                        f"loss={avg_loss:.4f} val_acc={val_acc:.4f} best_val_acc={best_val_acc:.4f}"
                    )
                else:
                    print(f"[Epoch {epoch:03d}/{cfg.epochs}] loss={avg_loss:.4f}")

        if has_val and best_state is not None:
            model.load_state_dict(best_state)
            model.to(self.device)

        return TrainingResult(
            model=model,
            num_classes=num_classes,
            device=self.device,
            history=self.history,
            best_val_accuracy=float(best_val_acc) if has_val else float("nan"),
        )

    def _evaluate_arrays(
        self,
        model: HyConExFromScratch,
        x: np.ndarray,
        y: np.ndarray,
    ) -> dict[str, Any]:
        model.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device)
            logits = model(x_tensor)
            proba = torch.softmax(logits, dim=1).cpu().numpy()
            y_pred = np.argmax(proba, axis=1)

        metrics: dict[str, Any] = {
            "accuracy": float(accuracy_score(y, y_pred)),
            "classification_report": classification_report(
                y, y_pred, output_dict=True, digits=4, zero_division=0
            ),
            "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
        }
        try:
            metrics["auroc_ovr"] = float(roc_auc_score(y, proba, multi_class="ovr"))
        except Exception as exc:  # noqa: BLE001
            metrics["auroc_ovr"] = None
            metrics["auroc_error"] = str(exc)
        return metrics

    def evaluate(
        self,
        X: np.ndarray | torch.Tensor,
        y: np.ndarray | torch.Tensor,
        *,
        counterfactuals: bool = True,
        cf_max_samples: int = 4000,
    ) -> dict[str, Any]:
        if self.model is None or self._num_classes is None:
            raise RuntimeError("Aucun modèle entraîné : appelez fit() d'abord.")

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        out: dict[str, Any] = self._evaluate_arrays(self.model, X, y)

        if counterfactuals:
            out["counterfactuals"] = self.evaluate_counterfactuals(
                X, y, max_samples=cf_max_samples
            )
        return out

    def evaluate_counterfactuals(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        max_samples: int = 4000,
    ) -> dict[str, Any]:
        if self.model is None or self._num_classes is None:
            raise RuntimeError("Aucun modèle entraîné : appelez fit() d'abord.")

        model = self.model
        num_classes = self._num_classes
        model.eval()
        idx = np.arange(x.shape[0])
        if x.shape[0] > max_samples:
            idx = np.random.choice(idx, size=max_samples, replace=False)

        x_sub = torch.tensor(x[idx], dtype=torch.float32, device=self.device)
        y_sub = torch.tensor(y[idx], dtype=torch.long, device=self.device)
        y_target = sample_alternative_targets(y_sub, num_classes)

        with torch.no_grad():
            x_cf = model.generate_counterfactual(x_sub, y_target)
            y_cf_pred = torch.argmax(model(x_cf), dim=1)

        validity = (y_cf_pred == y_target).float().mean().item()
        l1 = torch.norm(x_cf - x_sub, p=1, dim=1).mean().item()
        changed = ((x_cf - x_sub).abs() > 1e-3).float().sum(dim=1).mean().item()
        return {
            "validity_cf": float(validity),
            "proximity_l1_mean": float(l1),
            "changed_features_mean": float(changed),
            "n_evaluated": int(x_sub.shape[0]),
        }


def train(
    X_train: np.ndarray | torch.Tensor,
    y_train: np.ndarray | torch.Tensor,
    *,
    config: TrainConfig | None = None,
    X_val: np.ndarray | torch.Tensor | None = None,
    y_val: np.ndarray | torch.Tensor | None = None,
    X_test: np.ndarray | torch.Tensor | None = None,
    y_test: np.ndarray | torch.Tensor | None = None,
    device: str | torch.device | None = "auto",
    verbose: bool = True,
    evaluate_cf_on_test: bool = True,
) -> TrainingResult:
    """
    Fonction de convenance : crée un ``HyConExTrainer``, entraîne, puis évalue éventuellement le test.

    Exemple::

        from hyconex_from_scratch import train, TrainConfig

        res = train(X_train, y_train, X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test,
                    config=TrainConfig(epochs=50))
        print(res.test_metrics)
    """
    trainer = HyConExTrainer(config=config, device=device)
    result = trainer.fit(X_train, y_train, X_val=X_val, y_val=y_val, verbose=verbose)

    if X_test is not None and y_test is not None:
        result.test_metrics = trainer.evaluate(
            X_test, y_test, counterfactuals=evaluate_cf_on_test
        )
        if verbose and result.test_metrics is not None:
            acc = result.test_metrics["accuracy"]
            print(f"\nTest accuracy: {acc:.4f}")
            if evaluate_cf_on_test and "counterfactuals" in result.test_metrics:
                cf = result.test_metrics["counterfactuals"]
                print(
                    f"CF validity: {cf['validity_cf']:.4f} | "
                    f"CF proximity L1: {cf['proximity_l1_mean']:.4f}"
                )

    return result
