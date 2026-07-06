from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from nouveau_module.binarizer import TabularBinarizer
from nouveau_module.config import HybridDRConfig
from nouveau_module.main_rule_net import extract_rules, unpack_main_params
from nouveau_module.model import HybridDRNetModel


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _all_class_targets(num_classes: int, device: torch.device) -> torch.Tensor:
    return torch.arange(num_classes, device=device, dtype=torch.long)


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _class_weights_tensor(y: np.ndarray, num_classes: int, device: torch.device) -> torch.Tensor:
    counts = np.bincount(y.astype(np.int64), minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    weights = len(y) / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _is_imbalanced(y: np.ndarray, threshold: float = 0.85) -> bool:
    counts = np.bincount(y.astype(np.int64))
    if counts.size == 0:
        return False
    return float(counts.max() / counts.sum()) >= threshold


def _resolve_early_stop_metric(
    cfg: HybridDRConfig,
    y_train: np.ndarray,
) -> Literal["accuracy", "auroc", "deny_f1", "balanced_accuracy"]:
    if cfg.early_stop_metric in ("accuracy", "auroc", "deny_f1", "balanced_accuracy"):
        return cfg.early_stop_metric  # type: ignore[return-value]
    if _is_imbalanced(y_train):
        return "deny_f1"
    return "accuracy"


def _classification_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    weight: torch.Tensor | None,
    use_focal: bool,
    focal_gamma: float,
) -> torch.Tensor:
    if not use_focal:
        return F.cross_entropy(logits, targets, weight=weight)
    ce = F.cross_entropy(logits, targets, weight=weight, reduction="none")
    pt = torch.exp(-ce)
    return (((1.0 - pt) ** focal_gamma) * ce).mean()


def _val_stop_score(y_val: np.ndarray, y_pred: np.ndarray, proba: np.ndarray, metric: str) -> float:
    from sklearn.metrics import balanced_accuracy_score

    if metric == "accuracy":
        return float(accuracy_score(y_val, y_pred))
    if metric == "deny_f1":
        if len(np.unique(y_val)) < 2:
            return -1.0
        return float(f1_score(y_val, y_pred, pos_label=0, zero_division=0))
    if metric == "balanced_accuracy":
        return float(balanced_accuracy_score(y_val, y_pred))
    if proba.shape[1] == 2:
        try:
            return float(roc_auc_score(y_val, proba[:, 1]))
        except Exception:  # noqa: BLE001
            return -1.0
    try:
        return float(roc_auc_score(y_val, proba, multi_class="ovr"))
    except Exception:  # noqa: BLE001
        return -1.0


@dataclass
class TrainingResult:
    model: HybridDRNetModel
    binarizer: TabularBinarizer
    class_names: list[str]
    history: list[dict[str, float | int]]
    best_val_accuracy: float
    best_val_auroc: float
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
        enc = self.config.input_encoding
        if enc not in ("auto", "bipolar", "quantize"):
            enc = "auto"
        self.binarizer = TabularBinarizer(
            bins_per_feature=self.config.bins_per_feature,
            encoding=enc,  # type: ignore[arg-type]
        )
        self.model: HybridDRNetModel | None = None
        self.class_names: list[str] = []
        self._ce_weight: torch.Tensor | None = None

    def _build_model(self, input_dim_bin: int, num_classes: int) -> HybridDRNetModel:
        cfg = self.config
        model = HybridDRNetModel(
            input_dim_bin=input_dim_bin,
            num_classes=num_classes,
            num_rules=cfg.num_rules,
            hyper_hidden_dim=cfg.hyper_hidden_dim,
            cf_hidden_dim=cfg.cf_hidden_dim,
            temperature=cfg.temperature,
            tabresnet_n_blocks=cfg.tabresnet_n_blocks,
            tabresnet_dropout=cfg.tabresnet_dropout,
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
        resume: bool = False,
    ) -> TrainingResult:
        cfg = self.config
        if not resume:
            set_seed(cfg.seed)

        x_train_cont = np.asarray(x_train_cont, dtype=np.float32)
        y_train = np.asarray(y_train, dtype=np.int64)

        if resume:
            if self.model is None:
                raise RuntimeError("resume=True mais aucun modele entraîne (appelez fit() d'abord).")
            x_train_bin = self.binarizer.transform(x_train_cont)
            num_classes = self.model.num_classes
        else:
            x_train_bin = self.binarizer.fit_transform(x_train_cont, feature_names=feature_names)
            num_classes = int(np.max(y_train) + 1)
            self.class_names = class_names or [str(i) for i in range(num_classes)]

            if cfg.use_class_weights:
                self._ce_weight = _class_weights_tensor(y_train, num_classes, self.device)
            else:
                self._ce_weight = None

        stop_metric = _resolve_early_stop_metric(cfg, y_train)

        if resume:
            model = self.model
            model.train()
        else:
            model = self._build_model(x_train_bin.shape[1], num_classes)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

        train_ds = TensorDataset(
            torch.tensor(x_train_bin, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long),
        )
        sampler = None
        shuffle = True
        if cfg.use_weighted_sampler:
            counts = np.bincount(y_train.astype(np.int64), minlength=num_classes).astype(np.float64)
            counts = np.maximum(counts, 1.0)
            sample_w = 1.0 / counts[y_train.astype(np.int64)]
            sampler = WeightedRandomSampler(
                weights=torch.tensor(sample_w, dtype=torch.double),
                num_samples=len(sample_w),
                replacement=True,
            )
            shuffle = False
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            drop_last=False,
        )

        has_val = x_val_cont is not None and y_val is not None
        if has_val:
            x_val_cont = np.asarray(x_val_cont, dtype=np.float32)
            y_val = np.asarray(y_val, dtype=np.int64)
            x_val_bin = self.binarizer.transform(x_val_cont)

        best_state: dict[str, torch.Tensor] | None = None
        best_val_acc = -1.0
        best_val_auroc = -1.0
        best_score = -1.0
        history: list[dict[str, float | int]] = []

        for epoch in range(1, cfg.epochs + 1):
            model.train()
            running_loss = 0.0
            in_warmup = epoch <= cfg.cf_warmup_epochs
            cf_l = 0.0 if in_warmup else cfg.cf_lambda
            flip_l = 0.0 if in_warmup else cfg.flip_lambda

            for xb, yb in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)

                out = model(xb)
                ce = _classification_loss(
                    out.logits,
                    yb,
                    weight=self._ce_weight,
                    use_focal=cfg.use_focal_loss,
                    focal_gamma=cfg.focal_gamma,
                )

                w_rule, _, _, _ = unpack_main_params(
                    out.theta_main,
                    input_dim=model.input_dim_bin,
                    num_rules=model.num_rules,
                    num_classes=model.num_classes,
                )
                if w_rule.dim() == 3:
                    rule_sparse = w_rule.abs().mean()
                else:
                    rule_sparse = w_rule.abs().mean()

                loss = ce + cfg.rule_sparsity_lambda * rule_sparse

                if cf_l > 0.0 or flip_l > 0.0:
                    x_cf_all, logits_cf_all = model.generate_counterfactuals_all_classes(xb, yb)
                    class_ids = _all_class_targets(num_classes, xb.device).view(1, num_classes).expand(
                        xb.shape[0], -1
                    )
                    valid_mask = class_ids != yb.unsqueeze(1)

                    if cf_l > 0.0:
                        logits_cf_flat = logits_cf_all.reshape(-1, num_classes)
                        targets_flat = class_ids.reshape(-1)
                        mask_flat = valid_mask.reshape(-1)
                        ce_cf_all = F.cross_entropy(
                            logits_cf_flat,
                            targets_flat,
                            reduction="none",
                            weight=self._ce_weight,
                        )
                        loss = loss + cf_l * ce_cf_all[mask_flat].mean()

                    if flip_l > 0.0:
                        x_rep = xb.unsqueeze(1).expand(-1, num_classes, -1)
                        flip_all = (x_cf_all != x_rep).float().mean(dim=2)
                        loss = loss + flip_l * flip_all[valid_mask].mean()

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                optimizer.step()
                running_loss += loss.item() * xb.shape[0]

            avg_loss = running_loss / len(train_ds)

            train_acc = float("nan")
            val_acc = float("nan")
            val_loss_ce = float("nan")
            train_loss_ce = float("nan")
            val_auroc_f = float("nan")

            if has_val:
                val_metrics = self.evaluate_binary(x_val_bin, y_val, counterfactuals=False)  # type: ignore[arg-type]
                val_acc = float(val_metrics["accuracy"])
                val_auroc = val_metrics.get("auroc_ovr")
                val_auroc_f = float(val_auroc) if val_auroc is not None else -1.0
                with torch.no_grad():
                    model.eval()
                    x_vt = torch.tensor(x_val_bin, dtype=torch.float32, device=self.device)
                    y_vt = torch.tensor(y_val, dtype=torch.long, device=self.device)
                    v_logits = model.predict_logits(x_vt)
                    val_loss_ce = float(
                        F.cross_entropy(v_logits, y_vt, weight=self._ce_weight).item()
                    )
                    v_proba = torch.softmax(v_logits, dim=1).cpu().numpy()
                    v_pred = np.argmax(v_proba, axis=1)
                score = _val_stop_score(y_val, v_pred, v_proba, stop_metric)

                if score > best_score:
                    best_score = score
                    best_val_acc = val_acc
                    best_val_auroc = val_auroc_f
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                with torch.no_grad():
                    model.eval()

            with torch.no_grad():
                bs_eval = min(cfg.batch_size, 512)
                correct = 0
                total = 0
                ce_sum = 0.0
                for i in range(0, x_train_bin.shape[0], bs_eval):
                    xb_e = torch.tensor(
                        x_train_bin[i : i + bs_eval], dtype=torch.float32, device=self.device
                    )
                    yb_e = torch.tensor(
                        y_train[i : i + bs_eval], dtype=torch.long, device=self.device
                    )
                    logits_e = model.predict_logits(xb_e)
                    ce_sum += F.cross_entropy(logits_e, yb_e, weight=self._ce_weight, reduction="sum").item()
                    correct += int((logits_e.argmax(dim=1) == yb_e).sum().item())
                    total += int(yb_e.shape[0])
                train_acc = correct / max(total, 1)
                train_loss_ce = ce_sum / max(total, 1)

            history.append(
                {
                    "epoch": epoch,
                    "train_loss": float(avg_loss),
                    "train_loss_ce": float(train_loss_ce),
                    "val_loss": float(val_loss_ce) if has_val else float("nan"),
                    "train_accuracy": float(train_acc),
                    "val_accuracy": float(val_acc) if has_val else float("nan"),
                    "val_auroc": float(val_auroc_f) if has_val else float("nan"),
                    "best_val_accuracy": float(best_val_acc) if has_val else float("nan"),
                    "best_val_auroc": float(best_val_auroc) if has_val else float("nan"),
                }
            )

            if verbose:
                if has_val:
                    warmup_tag = " [warmup]" if in_warmup else ""
                    print(
                        f"[Epoch {epoch:03d}/{cfg.epochs}]{warmup_tag} "
                        f"loss={avg_loss:.4f} val_acc={val_acc:.4f} val_auroc={val_auroc_f:.4f} "
                        f"best({stop_metric})={best_score:.4f}"
                    )
                else:
                    print(f"[Epoch {epoch:03d}/{cfg.epochs}] loss={avg_loss:.4f}")

        if has_val and best_state is not None:
            model.load_state_dict(best_state)
            model.to(self.device)

        if verbose and has_val:
            print(f"  binarizer mode={self.binarizer.mode_} | early_stop={stop_metric}")

        return TrainingResult(
            model=model,
            binarizer=self.binarizer,
            class_names=self.class_names,
            history=history,
            best_val_accuracy=float(best_val_acc) if has_val else float("nan"),
            best_val_auroc=float(best_val_auroc) if has_val else float("nan"),
        )

    def predict_proba(self, x_cont: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Aucun modèle entraîné. Lance fit() d'abord.")
        x_bin = self.binarizer.transform(np.asarray(x_cont, dtype=np.float32))
        self.model.eval()
        with torch.no_grad():
            x_t = torch.tensor(x_bin, dtype=torch.float32, device=self.device)
            logits = self.model.predict_logits(x_t)
            return torch.softmax(logits, dim=1).cpu().numpy()

    def evaluate_binary(
        self,
        x_bin: np.ndarray,
        y: np.ndarray,
        *,
        counterfactuals: bool = True,
        grant_threshold: float | None = None,
    ) -> dict[str, Any]:
        if self.model is None:
            raise RuntimeError("Aucun modèle entraîné. Lance fit() d'abord.")

        self.model.eval()
        with torch.no_grad():
            x_t = torch.tensor(x_bin, dtype=torch.float32, device=self.device)
            logits = self.model.predict_logits(x_t)
            proba = torch.softmax(logits, dim=1).cpu().numpy()
            if grant_threshold is not None and proba.shape[1] == 2:
                from nouveau_module.binary_metrics import predict_with_grant_threshold

                y_pred = predict_with_grant_threshold(proba, grant_threshold)
            else:
                y_pred = np.argmax(proba, axis=1)

        metrics: dict[str, Any] = {
            "accuracy": float(accuracy_score(y, y_pred)),
            "classification_report": classification_report(y, y_pred, output_dict=True, digits=4, zero_division=0),
            "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
        }
        if grant_threshold is not None:
            metrics["grant_threshold"] = float(grant_threshold)
        try:
            if int(np.max(y) + 1) == 2:
                metrics["auroc_ovr"] = float(roc_auc_score(y, proba[:, 1]))
            else:
                metrics["auroc_ovr"] = float(roc_auc_score(y, proba, multi_class="ovr"))
        except Exception as exc:  # noqa: BLE001
            metrics["auroc_ovr"] = None
            metrics["auroc_error"] = str(exc)

        if counterfactuals:
            metrics["counterfactuals"] = self.evaluate_counterfactuals_binary(x_bin, y)
        return metrics

    def evaluate(
        self,
        x_cont: np.ndarray,
        y: np.ndarray,
        *,
        counterfactuals: bool = True,
        grant_threshold: float | None = None,
    ) -> dict[str, Any]:
        x_bin = self.binarizer.transform(np.asarray(x_cont, dtype=np.float32))
        return self.evaluate_binary(
            x_bin,
            np.asarray(y, dtype=np.int64),
            counterfactuals=counterfactuals,
            grant_threshold=grant_threshold,
        )

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
        flips_mat = (x_cf_all != x_rep).float().sum(dim=2)
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

        with torch.no_grad():
            x_probe = torch.randint(
                0,
                2,
                (min(32, 64), self.model.input_dim_bin),
                device=self.device,
                dtype=torch.float32,
            )
            x_probe = x_probe * 2.0 - 1.0
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
