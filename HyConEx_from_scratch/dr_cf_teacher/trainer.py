from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from dr_cf_teacher.config import CFTeacherDRConfig
from dr_cf_teacher.model import CFTeacherDRModel
from hyperlogic_pure.model import continuous_to_bipolar
from hyconex_from_scratch.trainer import sample_alternative_targets
from nouveau_module.binarizer import TabularBinarizer
from nouveau_module.config import HybridDRConfig as _StopCfg
from nouveau_module.main_rule_net import extract_rules, unpack_main_params
from nouveau_module.trainer import (
    _class_weights_tensor,
    _resolve_early_stop_metric,
    _val_stop_score,
    set_seed,
)


@dataclass
class CFTeacherTrainResult:
    model: CFTeacherDRModel
    binarizer: TabularBinarizer
    class_names: list[str]
    history: list[dict[str, float | int | str]]
    best_val_accuracy: float
    best_val_auroc: float
    teacher: CFTeacherDRModel | None = None


class CFTeacherDRTrainer:
    def __init__(self, config: CFTeacherDRConfig | None = None, *, device: str | None = "auto") -> None:
        self.config = config or CFTeacherDRConfig()
        if device is None or device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        enc = self.config.input_encoding
        if enc not in ("auto", "bipolar", "quantize"):
            enc = "bipolar"
        self.binarizer = TabularBinarizer(
            bins_per_feature=self.config.bins_per_feature,
            encoding=enc,  # type: ignore[arg-type]
        )
        self.model: CFTeacherDRModel | None = None
        self.teacher: CFTeacherDRModel | None = None
        self.class_names: list[str] = []
        self.feature_names: list[str] = []
        self._ce_weight: torch.Tensor | None = None

    def _build_model(self, input_dim: int, num_classes: int) -> CFTeacherDRModel:
        cfg = self.config
        model = CFTeacherDRModel(
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
        ).to(self.device)
        self.model = model
        return model

    def _rule_sparse(self, model: CFTeacherDRModel, theta_main: torch.Tensor) -> torch.Tensor:
        w_rule, _, _, _ = unpack_main_params(
            theta_main,
            input_dim=model.input_dim_bin,
            num_rules=model.num_rules,
            num_classes=model.num_classes,
        )
        return w_rule.abs().mean()

    def _cf_representation(self, xb: torch.Tensor) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("modele non initialise")
        if self.model.mode == "embed":
            emb = self.model.core.core.encode(xb)
            return continuous_to_bipolar(emb)
        return continuous_to_bipolar(xb)

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
    ) -> CFTeacherTrainResult:
        cfg = self.config
        set_seed(cfg.seed)

        x_train = np.asarray(x_train, dtype=np.float32)
        y_train = np.asarray(y_train, dtype=np.int64)
        high_dim = x_train.shape[1] > cfg.max_instance_dim
        if high_dim:
            x_train_bin = x_train
            self.binarizer.mode_ = "bipolar"
            self.binarizer.n_binary_features_ = x_train.shape[1]
            self.feature_names = feature_names or [f"oh_{i}" for i in range(x_train.shape[1])]
        else:
            x_train_bin = self.binarizer.fit_transform(x_train, feature_names=feature_names)
            self.feature_names = feature_names or list(self.binarizer.feature_names_)
        num_classes = int(np.max(y_train) + 1)
        self.class_names = class_names or [str(i) for i in range(num_classes)]
        if cfg.use_class_weights:
            self._ce_weight = _class_weights_tensor(y_train, num_classes, self.device)
        else:
            self._ce_weight = None

        model = self._build_model(x_train_bin.shape[1], num_classes)
        stop_metric = _resolve_early_stop_metric(_StopCfg(early_stop_metric=cfg.early_stop_metric), y_train)

        train_loader = DataLoader(
            TensorDataset(
                torch.tensor(x_train_bin, dtype=torch.float32),
                torch.tensor(y_train, dtype=torch.long),
            ),
            batch_size=cfg.batch_size,
            shuffle=True,
            drop_last=False,
        )

        has_val = x_val is not None and y_val is not None
        if has_val:
            x_val = np.asarray(x_val, dtype=np.float32)
            y_val = np.asarray(y_val, dtype=np.int64)
            x_val_bin = x_val if high_dim else self.binarizer.transform(x_val)

        history: list[dict[str, float | int | str]] = []
        best_state = None
        best_score = -1.0
        best_val_acc = -1.0
        best_val_auroc = -1.0

        # ----- Phase 1 : CF prédit -----
        model.set_phase("cf")
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )
        if verbose:
            print(f"[Phase 1] CF predict ({cfg.phase1_epochs} epochs)", flush=True)

        for epoch in range(1, cfg.phase1_epochs + 1):
            model.train()
            running = 0.0
            for xb, yb in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)

                # Phase 1 : la voie CF predit (CF vers y, puis regles sur x_cf)
                x_cf_y, logits_cf_y = model.core.generate_counterfactual(xb, yb)
                loss = cfg.cf_predict_lambda * F.cross_entropy(
                    logits_cf_y, yb, weight=self._ce_weight
                )

                y_alt = sample_alternative_targets(yb, num_classes)
                x_cf_alt, logits_cf_alt = model.core.generate_counterfactual(xb, y_alt)
                if cfg.cf_lambda > 0.0:
                    loss = loss + cfg.cf_lambda * F.cross_entropy(
                        logits_cf_alt, y_alt, weight=self._ce_weight
                    )
                if cfg.flip_lambda > 0.0:
                    x_rep = self._cf_representation(xb)
                    loss = loss + cfg.flip_lambda * (x_cf_alt != x_rep).float().mean()

                pack = model.core.forward(xb)
                loss = loss + cfg.rule_sparsity_lambda * self._rule_sparse(model, pack.theta_main)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                optimizer.step()
                running += loss.item() * xb.shape[0]

            avg_loss = running / len(train_loader.dataset)
            val_acc, val_auroc_f = (
                self._val_scores(model, x_val_bin, y_val, stop_metric, rules_only=False, cf_via_true_class=True)
                if has_val
                else (float("nan"), float("nan"))
            )
            if has_val:
                score = val_auroc_f if stop_metric == "auroc" else val_acc
                if score > best_score:
                    best_score = score
                    best_val_acc = val_acc
                    best_val_auroc = val_auroc_f
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

            history.append(
                {
                    "phase": "p1_cf",
                    "epoch": epoch,
                    "train_loss": float(avg_loss),
                    "val_accuracy": float(val_acc),
                    "val_auroc": float(val_auroc_f),
                }
            )
            if verbose:
                print(
                    f"  [P1 {epoch:03d}/{cfg.phase1_epochs}] loss={avg_loss:.4f} "
                    f"val_acc={val_acc:.4f} val_auroc={val_auroc_f:.4f}",
                    flush=True,
                )

        if best_state is not None:
            model.load_state_dict(best_state)
            model.to(self.device)

        self.teacher = model.clone_teacher()
        if verbose:
            print("[Phase 1 done] teacher CF fige pour phase 2", flush=True)

        best_score = -1.0
        best_state = None

        # ----- Phase 2 : règles apprennent du teacher CF -----
        model._freeze_tab_p2 = cfg.freeze_tab_phase2  # type: ignore[attr-defined]
        model.set_phase("rules")
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=cfg.lr_phase2,
            weight_decay=cfg.weight_decay,
        )
        if verbose:
            print(
                f"[Phase 2] rules distill ({cfg.phase2_epochs} epochs, "
                f"lambda={cfg.distill_lambda})",
                flush=True,
            )

        for epoch in range(1, cfg.phase2_epochs + 1):
            model.train()
            running = 0.0
            for xb, yb in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)

                logits_rules = model.forward_rules(xb)
                loss = cfg.rules_ce_lambda * F.cross_entropy(
                    logits_rules, yb, weight=self._ce_weight
                )

                if self.teacher is not None and cfg.distill_lambda > 0.0:
                    with torch.no_grad():
                        _, t_logits = self.teacher.core.generate_counterfactual(xb, yb)
                    t = cfg.distill_temperature
                    kd = F.kl_div(
                        F.log_softmax(logits_rules / t, dim=1),
                        F.softmax(t_logits / t, dim=1),
                        reduction="batchmean",
                    ) * (t * t)
                    loss = loss + cfg.distill_lambda * kd

                pack = model.core.forward(xb)
                loss = loss + cfg.rule_sparsity_lambda * self._rule_sparse(model, pack.theta_main)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                optimizer.step()
                running += loss.item() * xb.shape[0]

            avg_loss = running / len(train_loader.dataset)
            val_acc, val_auroc_f = self._val_scores(model, x_val_bin, y_val, stop_metric, rules_only=True) if has_val else (float("nan"), float("nan"))
            if has_val:
                score = val_auroc_f if stop_metric == "auroc" else val_acc
                if score > best_score:
                    best_score = score
                    best_val_acc = val_acc
                    best_val_auroc = val_auroc_f
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

            history.append(
                {
                    "phase": "p2_rules",
                    "epoch": epoch,
                    "train_loss": float(avg_loss),
                    "val_accuracy": float(val_acc),
                    "val_auroc": float(val_auroc_f),
                }
            )
            if verbose:
                print(
                    f"  [P2 {epoch:03d}/{cfg.phase2_epochs}] loss={avg_loss:.4f} "
                    f"val_acc={val_acc:.4f} val_auroc={val_auroc_f:.4f}",
                    flush=True,
                )

        if best_state is not None:
            model.load_state_dict(best_state)
            model.to(self.device)

        return CFTeacherTrainResult(
            model=model,
            binarizer=self.binarizer,
            class_names=self.class_names,
            history=history,
            best_val_accuracy=float(best_val_acc),
            best_val_auroc=float(best_val_auroc),
            teacher=self.teacher,
        )

    def _val_scores(
        self,
        model: CFTeacherDRModel,
        x_val_bin: np.ndarray,
        y_val: np.ndarray,
        stop_metric: str,
        *,
        rules_only: bool = False,
        max_samples: int = 800,
        cf_via_true_class: bool = False,
    ) -> tuple[float, float]:
        if len(y_val) > max_samples:
            idx = np.random.choice(len(y_val), size=max_samples, replace=False)
            x_val_bin = x_val_bin[idx]
            y_val = y_val[idx]
        model.eval()
        with torch.no_grad():
            x_t = torch.tensor(x_val_bin, dtype=torch.float32, device=self.device)
            if rules_only:
                logits = model.forward_rules(x_t)
            elif cf_via_true_class:
                y_t = torch.tensor(y_val, dtype=torch.long, device=self.device)
                _, logits = model.core.generate_counterfactual(x_t, y_t)
            else:
                logits = model.cf_predict_logits(x_t)
            proba = torch.softmax(logits, dim=1).cpu().numpy()
            pred = np.argmax(proba, axis=1)
        val_acc = float(accuracy_score(y_val, pred))
        val_auroc_f = _val_stop_score(y_val, pred, proba, "auroc")
        if stop_metric != "auroc":
            _ = val_auroc_f
        return val_acc, float(val_auroc_f)

    def predict_proba_cf_true_class(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        assert self.model is not None
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        x_in = x if self.model.mode == "embed" else self.binarizer.transform(x)
        self.model.eval()
        chunks: list[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, x_in.shape[0], 256):
                xb = torch.tensor(x_in[i : i + 256], dtype=torch.float32, device=self.device)
                yb = torch.tensor(y[i : i + 256], dtype=torch.long, device=self.device)
                _, logits = self.model.core.generate_counterfactual(xb, yb)
                chunks.append(torch.softmax(logits, dim=1).cpu().numpy())
        return np.vstack(chunks)

    def predict_proba(self, x: np.ndarray, *, rules_only: bool = True) -> np.ndarray:
        assert self.model is not None
        x = np.asarray(x, dtype=np.float32)
        if self.model.mode == "embed":
            x_in = x
        else:
            x_in = self.binarizer.transform(x)
        self.model.eval()
        chunks: list[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, x_in.shape[0], 256):
                xb = torch.tensor(x_in[i : i + 256], dtype=torch.float32, device=self.device)
                logits = (
                    self.model.forward_rules(xb)
                    if rules_only
                    else self.model.cf_predict_logits(xb)
                )
                chunks.append(torch.softmax(logits, dim=1).cpu().numpy())
        return np.vstack(chunks)

    def evaluate(self, x: np.ndarray, y: np.ndarray, *, counterfactuals: bool = True) -> dict[str, Any]:
        assert self.model is not None
        y = np.asarray(y, dtype=np.int64)
        proba_rules = self.predict_proba(x, rules_only=True)
        pred_rules = np.argmax(proba_rules, axis=1)
        out: dict[str, Any] = {
            "accuracy": float(accuracy_score(y, pred_rules)),
            "rules_only_accuracy": float(accuracy_score(y, pred_rules)),
        }
        try:
            if proba_rules.shape[1] == 2:
                out["auroc_ovr"] = float(roc_auc_score(y, proba_rules[:, 1]))
                out["rules_only_auroc"] = float(roc_auc_score(y, proba_rules[:, 1]))
            else:
                out["auroc_ovr"] = float(roc_auc_score(y, proba_rules, multi_class="ovr"))
                out["rules_only_auroc"] = float(out["auroc_ovr"])
        except Exception:  # noqa: BLE001
            out["auroc_ovr"] = None
            out["rules_only_auroc"] = None

        proba_cf = self.predict_proba_cf_true_class(x, y)
        pred_cf = np.argmax(proba_cf, axis=1)
        out["cf_predict_accuracy"] = float(accuracy_score(y, pred_cf))
        try:
            if proba_cf.shape[1] == 2:
                out["cf_predict_auroc"] = float(roc_auc_score(y, proba_cf[:, 1]))
            else:
                out["cf_predict_auroc"] = float(roc_auc_score(y, proba_cf, multi_class="ovr"))
        except Exception:  # noqa: BLE001
            out["cf_predict_auroc"] = None

        if counterfactuals:
            out["counterfactuals"] = self._evaluate_counterfactuals(x, y)
        return out

    def _evaluate_counterfactuals(self, x: np.ndarray, y: np.ndarray, *, max_samples: int = 2000) -> dict[str, float]:
        assert self.model is not None
        x = np.asarray(x, dtype=np.float32)
        if self.model.mode == "embed":
            x_bin = x
        else:
            x_bin = self.binarizer.transform(x)
        idx = np.arange(len(y))
        if len(idx) > max_samples:
            idx = np.random.choice(idx, size=max_samples, replace=False)
        x_sub = torch.tensor(x_bin[idx], dtype=torch.float32, device=self.device)
        y_sub = torch.tensor(y[idx], dtype=torch.long, device=self.device)
        num_classes = self.model.num_classes
        with torch.no_grad():
            x_cf_all, logits_cf_all = self.model.generate_counterfactuals_all_classes(x_sub, y_sub)
            class_ids = torch.arange(num_classes, device=self.device).view(1, num_classes).expand(
                x_sub.shape[0], -1
            )
            valid = class_ids != y_sub.unsqueeze(1)
            flat_logits = logits_cf_all.reshape(-1, num_classes)
            flat_targets = class_ids.reshape(-1)
            flat_mask = valid.reshape(-1)
            y_cf_pred = torch.argmax(flat_logits, dim=1)
            validity = (y_cf_pred[flat_mask] == flat_targets[flat_mask]).float().mean().item()
            x_rep = self._cf_representation(x_sub).unsqueeze(1).expand(-1, num_classes, -1)
            changed = (x_cf_all != x_rep).float().sum(dim=2).mean().item()
        return {
            "validity_cf": float(validity),
            "changed_bits_mean": float(changed),
            "n_evaluated": int(x_sub.shape[0]),
        }

    def export_rules(self, *, top_per_rule: int = 4, min_abs_weight: float = 0.001) -> list[dict]:
        assert self.model is not None
        dummy = torch.zeros(1, self.model.core.input_dim, device=self.device)
        pack = self.model.core.forward(dummy)
        theta = pack.theta_main[0:1]
        if self.model.mode == "embed":
            feat_names = [f"emb_{i}" for i in range(self.model.input_dim_bin)]
        else:
            feat_names = self.feature_names or [f"oh_{i}" for i in range(self.model.input_dim_bin)]
        w_rule, _, w_out, _ = unpack_main_params(
            theta,
            input_dim=self.model.input_dim_bin,
            num_rules=self.model.num_rules,
            num_classes=self.model.num_classes,
        )
        return extract_rules(
            w_rule,
            w_out,
            feat_names,
            self.class_names,
            top_per_rule=top_per_rule,
            min_abs_weight=min_abs_weight,
        )
