from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from hyperlogic_pure.config import PureDRConfig
from hyperlogic_pure.model import PureDRNetModel
from nouveau_module.main_rule_net import extract_rules, unpack_main_params
from nouveau_module.config import HybridDRConfig as _StopCfg
from nouveau_module.trainer import (
    _class_weights_tensor,
    _resolve_early_stop_metric,
    _val_stop_score,
    set_seed,
)


@dataclass
class PureTrainResult:
    model: PureDRNetModel
    class_names: list[str]
    feature_names: list[str]
    best_val_accuracy: float
    best_val_auroc: float
    history: list[dict[str, float | int]]


class PureDRNetTrainer:
    def __init__(
        self,
        config: PureDRConfig | None = None,
        *,
        device: str | torch.device | None = "auto",
    ) -> None:
        self.config = config or PureDRConfig()
        if device is None or device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.model: PureDRNetModel | None = None
        self.class_names: list[str] = []
        self.feature_names: list[str] = []
        self._ce_weight: torch.Tensor | None = None

    def _build(self, input_dim: int, num_classes: int) -> PureDRNetModel:
        cfg = self.config
        model = PureDRNetModel(
            input_dim,
            num_classes,
            num_rules=cfg.num_rules,
            cf_hidden_dim=cfg.cf_hidden_dim,
            hyper_hidden_dim=cfg.hyper_hidden_dim,
            temperature=cfg.temperature,
            tabresnet_n_blocks=cfg.tabresnet_n_blocks,
            tabresnet_dropout=cfg.tabresnet_dropout,
            max_instance_dim=cfg.max_instance_dim,
            ctx_hidden_dim=cfg.ctx_hidden_dim,
            ctx_modulation=cfg.ctx_modulation,
            embed_dim_high=cfg.embed_dim_high,
            cf_graft_scale=cfg.cf_graft_scale,
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

    def _freeze_drnet(self, model: PureDRNetModel) -> None:
        """Phase 2 : fige le classifieur DR-Net, entraîne uniquement la greffe CF."""
        for p in model.parameters():
            p.requires_grad = False
        core = model.core
        if model.mode == "embed":
            core.init_cf_modules()
            for p in model.parameters():
                p.requires_grad = False
            assert core.theta_cf_bias is not None
            core.theta_cf_bias.requires_grad = True
        elif model.mode == "global":
            core.init_cf_modules()
            for p in model.parameters():
                p.requires_grad = False
            assert core.cf_graft is not None and core.encoder is not None
            for p in core.cf_graft.parameters():
                p.requires_grad = True
            for p in core.encoder.parameters():
                p.requires_grad = True
        else:
            for p in core.hyper.cf_head.parameters():
                p.requires_grad = True
            for p in core.hyper.tab.parameters():
                p.requires_grad = True

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
        phase: str = "drnet",
        teacher_proba: np.ndarray | None = None,
    ) -> PureTrainResult:
        cfg = self.config
        if not resume:
            set_seed(cfg.seed)

        x_train = np.asarray(x_train, dtype=np.float32)
        y_train = np.asarray(y_train, dtype=np.int64)
        num_classes = int(np.max(y_train) + 1)
        self.class_names = class_names or [str(i) for i in range(num_classes)]
        self.feature_names = feature_names or [f"oh_{i}" for i in range(x_train.shape[1])]

        if resume:
            if self.model is None:
                raise RuntimeError("resume=True sans modèle existant")
            model = self.model
        else:
            model = self._build(x_train.shape[1], num_classes)
            if cfg.use_class_weights:
                self._ce_weight = _class_weights_tensor(y_train, num_classes, self.device)
            else:
                self._ce_weight = None

        if phase == "cf" and model.mode in ("global", "embed"):
            model.core.init_cf_modules()
            model.core.to(self.device)

        if phase == "cf" and cfg.freeze_drnet_phase2:
            self._freeze_drnet(model)

        lr = cfg.lr_phase2 if phase == "cf" else cfg.lr
        stop_metric = _resolve_early_stop_metric(
            _StopCfg(early_stop_metric=cfg.early_stop_metric),
            y_train,
        )

        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr,
            weight_decay=cfg.weight_decay,
        )

        use_distill = (
            teacher_proba is not None
            and phase == "drnet"
            and cfg.distill_lambda > 0.0
        )
        if use_distill:
            train_ds = TensorDataset(
                torch.tensor(x_train, dtype=torch.float32),
                torch.tensor(y_train, dtype=torch.long),
                torch.tensor(np.asarray(teacher_proba, dtype=np.float32)),
            )
        else:
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

        cf_l = cfg.cf_lambda_phase2 if phase == "cf" else cfg.cf_lambda
        flip_l = cfg.flip_lambda_phase2 if phase == "cf" else cfg.flip_lambda
        epochs = cfg.cf_epochs if phase == "cf" else cfg.epochs

        for epoch in range(1, epochs + 1):
            model.train()
            running = 0.0
            for batch in loader:
                if use_distill:
                    xb, yb, tb = batch
                    tb = tb.to(self.device)
                else:
                    xb, yb = batch
                    tb = None
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                out = model(xb)
                ce = F.cross_entropy(out.logits, yb, weight=self._ce_weight)
                if use_distill and tb is not None:
                    log_p = F.log_softmax(out.logits, dim=1)
                    loss_distill = F.kl_div(log_p, tb, reduction="batchmean")
                    if cfg.distill_only:
                        ce = cfg.distill_lambda * loss_distill
                    else:
                        ce = ce + cfg.distill_lambda * loss_distill

                theta_for_sparse = (
                    model.core.theta_main_bias.unsqueeze(0)
                    if model.mode in ("global", "embed")
                    else out.theta_main
                )
                w_rule, _, _, _ = unpack_main_params(
                    theta_for_sparse,
                    input_dim=model.input_dim_bin,
                    num_rules=model.num_rules,
                    num_classes=model.num_classes,
                )
                sparse = w_rule.abs().mean()
                loss = ce + cfg.rule_sparsity_lambda * sparse

                if cf_l > 0.0 or flip_l > 0.0:
                    x_cf_all, logits_cf_all = model.generate_counterfactuals_all_classes(xb, yb)
                    class_ids = torch.arange(num_classes, device=xb.device).view(1, num_classes).expand(
                        xb.shape[0], -1
                    )
                    valid = class_ids != yb.unsqueeze(1)
                    if cf_l > 0.0:
                        flat_logits = logits_cf_all.reshape(-1, num_classes)
                        flat_targets = class_ids.reshape(-1)
                        flat_mask = valid.reshape(-1)
                        ce_cf = F.cross_entropy(
                            flat_logits, flat_targets, reduction="none", weight=self._ce_weight
                        )
                        loss = loss + cf_l * ce_cf[flat_mask].mean()
                    if flip_l > 0.0:
                        from hyperlogic_pure.model import continuous_to_bipolar

                        if model.mode == "embed":
                            emb = model.core.encode(xb)
                            x_rep = continuous_to_bipolar(emb).unsqueeze(1).expand(-1, num_classes, -1)
                        else:
                            x_rep = continuous_to_bipolar(xb).unsqueeze(1).expand(-1, num_classes, -1)
                        flip = (x_cf_all != x_rep).float().mean(dim=2)
                        loss = loss + flip_l * flip[valid].mean()

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
                    "phase": phase,
                    "train_loss": float(avg_loss),
                    "val_accuracy": float(val_acc),
                    "val_auroc": float(val_auroc_f),
                }
            )
            if verbose:
                print(
                    f"[{phase} Epoch {epoch:03d}/{epochs}] loss={avg_loss:.4f} "
                    f"val_acc={val_acc:.4f} val_auroc={val_auroc_f:.4f} best({stop_metric})={best_score:.4f}",
                    flush=True,
                )

        if has_val and best_state is not None:
            model.load_state_dict(best_state)
            model.to(self.device)

        return PureTrainResult(
            model=model,
            class_names=self.class_names,
            feature_names=self.feature_names,
            best_val_accuracy=best_val_acc,
            best_val_auroc=best_val_auroc,
            history=history,
        )

    @torch.no_grad()
    def predict_proba(self, x: np.ndarray, *, batch_size: int = 128) -> np.ndarray:
        assert self.model is not None
        self.model.eval()
        x = np.asarray(x, dtype=np.float32)
        chunks: list[np.ndarray] = []
        for start in range(0, x.shape[0], batch_size):
            xb = x[start : start + batch_size]
            x_t = torch.tensor(xb, dtype=torch.float32, device=self.device)
            logits = self.model.predict_logits(x_t)
            chunks.append(torch.softmax(logits, dim=1).cpu().numpy())
        return np.concatenate(chunks, axis=0)

    @torch.no_grad()
    def evaluate(self, x: np.ndarray, y: np.ndarray, *, counterfactuals: bool = True) -> dict[str, Any]:
        assert self.model is not None
        self.model.eval()
        y = np.asarray(y, dtype=np.int64)
        proba = self.predict_proba(x)
        pred = np.argmax(proba, axis=1)
        metrics: dict[str, Any] = {"accuracy": float(accuracy_score(y, pred))}
        if len(np.unique(y)) >= 2 and proba.shape[1] == 2:
            metrics["auroc_ovr"] = float(roc_auc_score(y, proba[:, 1]))
        elif len(np.unique(y)) >= 2:
            metrics["auroc_ovr"] = float(roc_auc_score(y, proba, multi_class="ovr"))

        if counterfactuals:
            can_cf = True
            if self.model.mode == "embed" and getattr(self.model.core, "theta_cf_bias", None) is None:
                can_cf = False
            if self.model.mode == "global" and getattr(self.model.core, "cf_graft", None) is None:
                can_cf = False
            if not can_cf:
                metrics["counterfactuals"] = {"validity_cf": 0.0, "skipped": True}
                return metrics
            x_t = torch.tensor(np.asarray(x, dtype=np.float32), device=self.device)
            y_t = torch.tensor(y, dtype=torch.long, device=self.device)
            valid = 0
            n = min(64, x_t.shape[0])
            for i in range(n):
                y_i = int(y_t[i].item())
                targets = [c for c in range(self.model.num_classes) if c != y_i]
                if not targets:
                    continue
                y_tgt = targets[0]
                x_cf, logits_cf = self.model.generate_counterfactual(
                    x_t[i : i + 1], torch.tensor([y_tgt], device=self.device)
                )
                pred_cf = int(logits_cf.argmax(dim=1).item())
                if pred_cf == y_tgt:
                    valid += 1
            metrics["counterfactuals"] = {"validity_cf": valid / max(n, 1)}
        return metrics

    def export_rules(self, top_per_rule: int = 4, min_abs_weight: float = 0.001) -> list[dict]:
        assert self.model is not None
        model = self.model
        if model.mode == "embed":
            feat_names = [f"emb_{i}" for i in range(model.core.embed_dim)]
            theta = model.core.theta_main_bias
        elif model.mode == "global":
            theta = model.core.theta_main_bias
            feat_names = self.feature_names
        else:
            dummy = torch.zeros(1, model.input_dim_bin, device=self.device)
            out = model.core(dummy)
            theta = out.theta_main.mean(dim=0) if out.theta_main.dim() == 2 else out.theta_main
            feat_names = self.feature_names

        w_rule, _, w_out, _ = unpack_main_params(
            theta.unsqueeze(0) if theta.dim() == 1 else theta,
            input_dim=model.input_dim_bin,
            num_rules=model.num_rules,
            num_classes=model.num_classes,
        )
        if w_rule.dim() == 3:
            w_rule = w_rule.mean(dim=0)
            w_out = w_out.mean(dim=0)
        return extract_rules(
            w_rule,
            w_out,
            feat_names,
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
                "mode": self.model.mode,
            },
            path,
        )

    @classmethod
    def load_checkpoint(cls, path: Path | str, *, device: str | torch.device | None = "auto") -> "PureDRNetTrainer":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        cfg = PureDRConfig(**payload["config"])
        trainer = cls(cfg, device=device)
        trainer.class_names = list(payload["class_names"])
        trainer.feature_names = list(payload["feature_names"])
        model = PureDRNetModel(
            int(payload["input_dim"]),
            int(payload["num_classes"]),
            num_rules=cfg.num_rules,
            cf_hidden_dim=cfg.cf_hidden_dim,
            hyper_hidden_dim=cfg.hyper_hidden_dim,
            temperature=cfg.temperature,
            tabresnet_n_blocks=cfg.tabresnet_n_blocks,
            tabresnet_dropout=cfg.tabresnet_dropout,
            max_instance_dim=cfg.max_instance_dim,
            ctx_hidden_dim=cfg.ctx_hidden_dim,
            ctx_modulation=cfg.ctx_modulation,
            embed_dim_high=cfg.embed_dim_high,
            cf_graft_scale=cfg.cf_graft_scale,
        )
        model.load_state_dict(payload["state_dict"])
        trainer.model = model.to(trainer.device)
        trainer.model.eval()
        return trainer
