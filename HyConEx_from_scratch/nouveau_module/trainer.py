from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from nouveau_module.binarizer import TabularBinarizer
from nouveau_module.config import HybridDRConfig
from nouveau_module.main_rule_net import extract_rules, unpack_main_params
from nouveau_module.model import HybridDRNetModel


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sample_alternative_targets(y: torch.Tensor, num_classes: int) -> torch.Tensor:
    noise = torch.randint(low=1, high=num_classes, size=y.shape, device=y.device)
    return (y + noise) % num_classes


def _all_class_targets(num_classes: int, device: torch.device) -> torch.Tensor:
    return torch.arange(num_classes, device=device, dtype=torch.long)


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


@dataclass
class TrainingResult:
    model: HybridDRNetModel
    binarizer: TabularBinarizer
    class_names: list[str]
    history: list[dict[str, float | int]]
    best_val_accuracy: float
    test_metrics: dict[str, Any] | None = None


class HybridDRTrainer:
    def __init__(
        self,
        config: HybridDRConfig | None = None,
        *,
        device: str | torch.device | None = "auto",
    ) -> None:
        self.config = config or HybridDRConfig()
        self.device = _resolve_device(device)
        self.binarizer = TabularBinarizer(bins_per_feature=self.config.bins_per_feature)
        self.model: HybridDRNetModel | None = None
        self.class_names: list[str] = []

    def _build_model(self, input_dim_bin: int, num_classes: int) -> HybridDRNetModel:
        cfg = self.config
        model = HybridDRNetModel(
            input_dim_bin=input_dim_bin,
            num_classes=num_classes,
            num_rules=cfg.num_rules,
            hyper_hidden_dim=cfg.hyper_hidden_dim,
            cf_hidden_dim=cfg.cf_hidden_dim,
            temperature=cfg.temperature,
        ).to(self.device)
        self.model = model
        return model

    def fit(
        self,
        x_train_cont: np.ndarray,
        y_train: np.ndarray,
        x_val_cont: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        *,
        feature_names: list[str] | None = None,
        class_names: list[str] | None = None,
        verbose: bool = True,
    ) -> TrainingResult:
        cfg = self.config
        set_seed(cfg.seed)

        x_train_cont = np.asarray(x_train_cont, dtype=np.float32)
        y_train = np.asarray(y_train, dtype=np.int64)

        x_train_bin = self.binarizer.fit_transform(x_train_cont, feature_names=feature_names)
        num_classes = int(np.max(y_train) + 1)
        self.class_names = class_names or [str(i) for i in range(num_classes)]

        model = self._build_model(x_train_bin.shape[1], num_classes)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

        train_ds = TensorDataset(
            torch.tensor(x_train_bin, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long),
        )
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)

        has_val = x_val_cont is not None and y_val is not None
        if has_val:
            x_val_cont = np.asarray(x_val_cont, dtype=np.float32)
            y_val = np.asarray(y_val, dtype=np.int64)
            x_val_bin = self.binarizer.transform(x_val_cont)

        best_state: dict[str, torch.Tensor] | None = None
        best_val_acc = -1.0
        history: list[dict[str, float | int]] = []

        for epoch in range(1, cfg.epochs + 1):
            model.train()
            running_loss = 0.0

            for xb, yb in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)

                out = model(xb)
                ce = F.cross_entropy(out.logits, yb)

                x_cf_all, logits_cf_all = model.generate_counterfactuals_all_classes(xb, yb)
                class_ids = _all_class_targets(num_classes, xb.device).view(1, num_classes).expand(xb.shape[0], -1)
                valid_mask = class_ids != yb.unsqueeze(1)

                logits_cf_flat = logits_cf_all.reshape(-1, num_classes)
                targets_flat = class_ids.reshape(-1)
                mask_flat = valid_mask.reshape(-1)
                ce_cf_all = F.cross_entropy(logits_cf_flat, targets_flat, reduction="none")
                ce_cf = ce_cf_all[mask_flat].mean()

                x_rep = xb.unsqueeze(1).expand(-1, num_classes, -1)
                flip_all = (x_cf_all - x_rep).abs().mean(dim=2)
                flip_cost = flip_all[valid_mask].mean()

                w_rule, _, _, _ = unpack_main_params(
                    out.theta_main,
                    input_dim=model.input_dim_bin,
                    num_rules=model.num_rules,
                    num_classes=model.num_classes,
                )
                rule_sparse = w_rule.abs().mean()

                loss = (
                    ce
                    + cfg.cf_lambda * ce_cf
                    + cfg.flip_lambda * flip_cost
                    + cfg.rule_sparsity_lambda * rule_sparse
                )

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                optimizer.step()
                running_loss += loss.item() * xb.shape[0]

            avg_loss = running_loss / len(train_ds)

            if has_val:
                val_metrics = self.evaluate_binary(x_val_bin, y_val, counterfactuals=False)  # type: ignore[arg-type]
                val_acc = float(val_metrics["accuracy"])
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                val_acc = float("nan")

            history.append(
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
            binarizer=self.binarizer,
            class_names=self.class_names,
            history=history,
            best_val_accuracy=float(best_val_acc) if has_val else float("nan"),
        )

    def evaluate_binary(self, x_bin: np.ndarray, y: np.ndarray, *, counterfactuals: bool = True) -> dict[str, Any]:
        if self.model is None:
            raise RuntimeError("Aucun modèle entraîné. Lance fit() d'abord.")

        self.model.eval()
        with torch.no_grad():
            x_t = torch.tensor(x_bin, dtype=torch.float32, device=self.device)
            logits = self.model.predict_logits(x_t)
            proba = torch.softmax(logits, dim=1).cpu().numpy()
            y_pred = np.argmax(proba, axis=1)

        metrics: dict[str, Any] = {
            "accuracy": float(accuracy_score(y, y_pred)),
            "classification_report": classification_report(y, y_pred, output_dict=True, digits=4, zero_division=0),
            "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
        }
        try:
            metrics["auroc_ovr"] = float(roc_auc_score(y, proba, multi_class="ovr"))
        except Exception as exc:  # noqa: BLE001
            metrics["auroc_ovr"] = None
            metrics["auroc_error"] = str(exc)

        if counterfactuals:
            metrics["counterfactuals"] = self.evaluate_counterfactuals_binary(x_bin, y)
        return metrics

    def evaluate(self, x_cont: np.ndarray, y: np.ndarray, *, counterfactuals: bool = True) -> dict[str, Any]:
        x_bin = self.binarizer.transform(np.asarray(x_cont, dtype=np.float32))
        return self.evaluate_binary(x_bin, np.asarray(y, dtype=np.int64), counterfactuals=counterfactuals)

    def evaluate_counterfactuals_binary(self, x_bin: np.ndarray, y: np.ndarray) -> dict[str, Any]:
        if self.model is None:
            raise RuntimeError("Aucun modèle entraîné. Lance fit() d'abord.")

        self.model.eval()
        x_t = torch.tensor(x_bin, dtype=torch.float32, device=self.device)
        y_t = torch.tensor(y, dtype=torch.long, device=self.device)

        with torch.no_grad():
            x_cf_all, logits_cf_all = self.model.generate_counterfactuals_all_classes(x_t, y_t)
            y_cf_pred_all = torch.argmax(logits_cf_all, dim=2)

        num_classes = self.model.num_classes
        class_ids = _all_class_targets(num_classes, x_t.device).view(1, num_classes).expand(x_t.shape[0], -1)
        valid_mask = class_ids != y_t.unsqueeze(1)

        validity_mat = (y_cf_pred_all == class_ids).float()
        validity = validity_mat[valid_mask].mean().item()

        x_rep = x_t.unsqueeze(1).expand(-1, num_classes, -1)
        flips_mat = (x_cf_all - x_rep).abs().sum(dim=2)
        flips = flips_mat[valid_mask].mean().item()

        x_cf_all_np = x_cf_all.detach().cpu().numpy().reshape(-1, x_t.shape[1])
        x_cf_cont_all = self.binarizer.binary_to_continuous(x_cf_all_np).reshape(x_t.shape[0], num_classes, -1)
        x_orig_cont = self.binarizer.binary_to_continuous(x_t.cpu().numpy())
        x_orig_cont_rep = np.repeat(x_orig_cont[:, None, :], num_classes, axis=1)
        l1_mat = np.abs(x_cf_cont_all - x_orig_cont_rep).sum(axis=2)
        l1_cont = float(l1_mat[valid_mask.cpu().numpy()].mean())

        return {
            "validity_cf": float(validity),
            "changed_bits_mean": float(flips),
            "proximity_l1_cont_mean": float(l1_cont),
            "targets_per_sample": int(max(0, num_classes - 1)),
        }

    def export_rules(self, top_per_rule: int = 4, min_abs_weight: float = 0.05) -> list[dict]:
        if self.model is None:
            raise RuntimeError("Aucun modèle entraîné. Lance fit() d'abord.")

        # Utilise un batch synthétique uniforme pour extraire theta globalement stable.
        x_probe = torch.full((16, self.model.input_dim_bin), 0.5, device=self.device)
        with torch.no_grad():
            theta_main, _ = self.model.hyper(x_probe)
        w_rule, _, w_out, _ = unpack_main_params(
            theta_main,
            input_dim=self.model.input_dim_bin,
            num_rules=self.model.num_rules,
            num_classes=self.model.num_classes,
        )

        return extract_rules(
            w_rule,
            w_out,
            binary_feature_names=self.binarizer.binary_feature_names(),
            class_names=self.class_names,
            top_per_rule=top_per_rule,
            min_abs_weight=min_abs_weight,
        )
