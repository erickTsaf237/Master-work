from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from hyconex_hyperlogic.config import HybridConfig
from hyconex_hyperlogic.model import HyConExHyperLogicModel
from nouveau_module.main_rule_net import extract_rules, unpack_main_params
from nouveau_module.trainer import _class_weights_tensor, _resolve_early_stop_metric, _val_stop_score, set_seed


@dataclass
class TrainResult:
    model: HyConExHyperLogicModel
    class_names: list[str]
    feature_names: list[str]
    best_val_accuracy: float
    best_val_auroc: float
    history: list[dict[str, float | int]]


class HyConExHyperLogicTrainer:
    def __init__(
        self,
        config: HybridConfig | None = None,
        *,
        device: str | torch.device | None = "auto",
    ) -> None:
        self.config = config or HybridConfig()
        if device is None or device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.model: HyConExHyperLogicModel | None = None
        self.class_names: list[str] = []
        self.feature_names: list[str] = []
        self._ce_weight: torch.Tensor | None = None

    def _build(self, input_dim: int, num_classes: int) -> HyConExHyperLogicModel:
        cfg = self.config
        model = HyConExHyperLogicModel(
            input_dim,
            num_classes,
            embed_dim=cfg.embed_dim,
            num_rules=cfg.num_rules,
            cf_hidden_dim=cfg.cf_hidden_dim,
            temperature=cfg.temperature,
            linear_weight=cfg.linear_weight,
            rule_weight=cfg.rule_weight,
        )
        try:
            model = model.to(self.device)
        except torch.cuda.OutOfMemoryError:
            if self.device.type == "cuda":
                print("  CUDA OOM -> repli CPU", flush=True)
                self.device = torch.device("cpu")
                model = model.to(self.device)
            else:
                raise
        self.model = model
        return model

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        *,
        feature_names: list[str] | None = None,
        class_names: list[str] | None = None,
        verbose: bool = True,
        resume: bool = False,
    ) -> TrainResult:
        cfg = self.config
        if not resume:
            set_seed(cfg.seed)

        x_train = np.asarray(x_train, dtype=np.float32)
        y_train = np.asarray(y_train, dtype=np.int64)
        num_classes = int(np.max(y_train) + 1)
        self.class_names = class_names or [str(i) for i in range(num_classes)]
        self.feature_names = feature_names or [f"f{i}" for i in range(x_train.shape[1])]

        if resume:
            if self.model is None:
                raise RuntimeError("resume=True sans modele existant")
            model = self.model
        else:
            model = self._build(x_train.shape[1], num_classes)
            if cfg.use_class_weights:
                self._ce_weight = _class_weights_tensor(y_train, num_classes, self.device)
            else:
                self._ce_weight = None

        stop_metric = _resolve_early_stop_metric(cfg, y_train)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

        train_ds = TensorDataset(
            torch.tensor(x_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long),
        )
        loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)

        has_val = x_val is not None and y_val is not None
        if has_val:
            x_val = np.asarray(x_val, dtype=np.float32)
            y_val = np.asarray(y_val, dtype=np.int64)

        best_state = None
        best_score = -1.0
        best_val_acc = -1.0
        best_val_auroc = -1.0
        history: list[dict[str, float | int]] = []

        for epoch in range(1, cfg.epochs + 1):
            model.train()
            running = 0.0
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                out = model(xb)
                ce = F.cross_entropy(out.logits, yb, weight=self._ce_weight)

                w_rule, _, _, _ = unpack_main_params(
                    model.theta_bias.unsqueeze(0),
                    input_dim=model.embed_dim,
                    num_rules=model.num_rules,
                    num_classes=model.num_classes,
                )
                sparse = w_rule.abs().mean()
                loss = ce + cfg.rule_sparsity_lambda * sparse

                if cfg.cf_lambda > 0.0 or cfg.flip_lambda > 0.0:
                    x_cf_all, logits_cf_all = model.generate_counterfactuals_all_classes(xb, yb)
                    class_ids = torch.arange(num_classes, device=xb.device).view(1, num_classes).expand(
                        xb.shape[0], -1
                    )
                    valid = class_ids != yb.unsqueeze(1)
                    if cfg.cf_lambda > 0.0:
                        flat_logits = logits_cf_all.reshape(-1, num_classes)
                        flat_targets = class_ids.reshape(-1)
                        flat_mask = valid.reshape(-1)
                        ce_cf = F.cross_entropy(flat_logits, flat_targets, reduction="none", weight=self._ce_weight)
                        loss = loss + cfg.cf_lambda * ce_cf[flat_mask].mean()
                    if cfg.flip_lambda > 0.0:
                        x_rep = xb.unsqueeze(1).expand(-1, num_classes, -1)
                        flip = (x_cf_all != x_rep).float().mean(dim=2)
                        loss = loss + cfg.flip_lambda * flip[valid].mean()

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                optimizer.step()
                running += loss.item() * xb.shape[0]

            avg_loss = running / len(train_ds)

            if has_val:
                metrics = self.evaluate(x_val, y_val, counterfactuals=False)
                val_acc = float(metrics["accuracy"])
                val_auroc = metrics.get("auroc_ovr")
                val_auroc_f = float(val_auroc) if val_auroc is not None else -1.0
                proba = self.predict_proba(x_val)
                pred = np.argmax(proba, axis=1)
                score = _val_stop_score(y_val, pred, proba, stop_metric)
                if score > best_score:
                    best_score = score
                    best_val_acc = val_acc
                    best_val_auroc = val_auroc_f
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                val_acc = float("nan")
                val_auroc_f = float("nan")

            history.append(
                {
                    "epoch": epoch,
                    "train_loss": float(avg_loss),
                    "val_accuracy": float(val_acc),
                    "val_auroc": float(val_auroc_f),
                }
            )
            if verbose:
                print(
                    f"[Epoch {epoch:03d}/{cfg.epochs}] loss={avg_loss:.4f} "
                    f"val_acc={val_acc:.4f} val_auroc={val_auroc_f:.4f} best({stop_metric})={best_score:.4f}",
                    flush=True,
                )

        if has_val and best_state is not None:
            model.load_state_dict(best_state)
            model.to(self.device)

        return TrainResult(
            model=model,
            class_names=self.class_names,
            feature_names=self.feature_names,
            best_val_accuracy=best_val_acc,
            best_val_auroc=best_val_auroc,
            history=history,
        )

    @torch.no_grad()
    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        assert self.model is not None
        self.model.eval()
        x_t = torch.tensor(np.asarray(x, dtype=np.float32), device=self.device)
        logits = self.model.predict_logits(x_t)
        return torch.softmax(logits, dim=1).cpu().numpy()

    @torch.no_grad()
    def evaluate(self, x: np.ndarray, y: np.ndarray, *, counterfactuals: bool = True) -> dict[str, Any]:
        assert self.model is not None
        self.model.eval()
        y = np.asarray(y, dtype=np.int64)
        proba = self.predict_proba(x)
        pred = np.argmax(proba, axis=1)
        metrics: dict[str, Any] = {
            "accuracy": float(accuracy_score(y, pred)),
        }
        if len(np.unique(y)) >= 2 and proba.shape[1] == 2:
            metrics["auroc_ovr"] = float(roc_auc_score(y, proba[:, 1]))
        elif len(np.unique(y)) >= 2:
            metrics["auroc_ovr"] = float(roc_auc_score(y, proba, multi_class="ovr"))

        if counterfactuals:
            x_t = torch.tensor(np.asarray(x, dtype=np.float32), device=self.device)
            y_t = torch.tensor(y, dtype=torch.long, device=self.device)
            valid = 0
            changed = 0.0
            prox = 0.0
            n = min(64, x_t.shape[0])
            for i in range(n):
                y_i = int(y_t[i].item())
                targets = [c for c in range(self.model.num_classes) if c != y_i]
                if not targets:
                    continue
                y_tgt = targets[0]
                x_cf = self.model.generate_counterfactual(x_t[i : i + 1], torch.tensor([y_tgt], device=self.device))
                pred_cf = int(self.model.predict_logits(x_cf).argmax(dim=1).item())
                if pred_cf == y_tgt:
                    valid += 1
                changed += float((x_cf != x_t[i : i + 1]).float().mean().item())
                prox += float((x_cf - x_t[i : i + 1]).abs().mean().item())
            metrics["counterfactuals"] = {
                "validity_cf": valid / max(n, 1),
                "changed_bits_mean": changed / max(n, 1),
                "proximity_l1_cont_mean": prox / max(n, 1),
            }
        return metrics

    def export_rules(self, top_per_rule: int = 4, min_abs_weight: float = 0.05) -> list[dict]:
        assert self.model is not None
        w_rule, _, w_out, _ = unpack_main_params(
            self.model.theta_bias.unsqueeze(0),
            input_dim=self.model.embed_dim,
            num_rules=self.model.num_rules,
            num_classes=self.model.num_classes,
        )
        rule_feat_names = [f"emb_{i}" for i in range(self.model.embed_dim)]
        return extract_rules(
            w_rule.squeeze(0),
            w_out.squeeze(0),
            rule_feat_names,
            self.class_names,
            top_per_rule=top_per_rule,
            min_abs_weight=min_abs_weight,
        )

    def save_checkpoint(self, path: Path | str) -> None:
        assert self.model is not None
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "config": asdict(self.config),
                "class_names": self.class_names,
                "feature_names": self.feature_names,
                "input_dim": self.model.input_dim,
                "num_classes": self.model.num_classes,
            },
            path,
        )

    @classmethod
    def load_checkpoint(
        cls,
        path: Path | str,
        *,
        device: str | torch.device | None = "auto",
    ) -> "HyConExHyperLogicTrainer":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        cfg = HybridConfig(**payload["config"])
        trainer = cls(cfg, device=device)
        trainer.class_names = list(payload["class_names"])
        trainer.feature_names = list(payload["feature_names"])
        model = HyConExHyperLogicModel(
            int(payload["input_dim"]),
            int(payload["num_classes"]),
            embed_dim=cfg.embed_dim,
            num_rules=cfg.num_rules,
            cf_hidden_dim=cfg.cf_hidden_dim,
            temperature=cfg.temperature,
            linear_weight=cfg.linear_weight,
            rule_weight=cfg.rule_weight,
        )
        model.load_state_dict(payload["state_dict"])
        trainer.model = model.to(trainer.device)
        trainer.model.eval()
        return trainer
