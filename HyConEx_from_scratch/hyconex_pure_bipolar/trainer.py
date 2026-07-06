from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from hyconex_from_scratch.trainer import HyConExTrainer, sample_alternative_targets, set_seed
from hyconex_pure_bipolar.bipolar import bipolar_to_continuous, continuous_to_bipolar
from hyconex_pure_bipolar.config import BipolarRulesConfig
from hyconex_pure_bipolar.model import HyConExBipolarRulesModel
from nouveau_module.main_rule_net import extract_rules, unpack_main_params


class HyConExBipolarRulesTrainer(HyConExTrainer):
    def __init__(self, config: BipolarRulesConfig | None = None, *, device=None) -> None:
        super().__init__(config=config or BipolarRulesConfig(), device=device)
        self.config: BipolarRulesConfig = self.config  # type: ignore[assignment]
        self.class_names: list[str] = []
        self.feature_names: list[str] = []
        self.rule_feature_names: list[str] = []

    def _ensure_model(self, input_dim: int, num_classes: int) -> HyConExBipolarRulesModel:
        cfg: BipolarRulesConfig = self.config
        self._num_classes = num_classes
        self.model = HyConExBipolarRulesModel(
            input_dim=input_dim,
            num_classes=num_classes,
            latent_dim=cfg.latent_dim,
            hidden_dim=cfg.hidden_dim,
            num_rules=cfg.num_rules,
            temperature=cfg.temperature,
            hyper_weight=cfg.hyper_weight,
            rule_weight=cfg.rule_weight,
            ctx_modulation=cfg.ctx_modulation,
            max_drnet_input_dim=cfg.max_drnet_input_dim,
        ).to(self.device)
        if self.model.rules_on_input:
            self.rule_feature_names = list(self.feature_names) or [f"oh_{i}" for i in range(input_dim)]
        else:
            self.rule_feature_names = [f"z_{i}" for i in range(cfg.latent_dim)]
        return self.model

    @staticmethod
    def _to_bipolar_np(x: np.ndarray) -> np.ndarray:
        return np.asarray(continuous_to_bipolar(x), dtype=np.float32)

    def fit(self, X_train, y_train, X_val=None, y_val=None, *, verbose: bool = True):
        cfg: BipolarRulesConfig = self.config
        set_seed(cfg.seed)

        X_train = self._to_bipolar_np(X_train)
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
            X_val = self._to_bipolar_np(X_val)
            y_val = np.asarray(y_val, dtype=np.int64)

        best_val_acc = -1.0
        best_state = None
        self.history = []

        def _epoch_metrics() -> tuple[float, float, float, float]:
            """train_acc, train_loss_ce, val_acc, val_loss_ce"""
            model.eval()
            train_correct = 0
            train_total = 0
            train_ce = 0.0
            with torch.no_grad():
                for xb, yb in train_loader:
                    xb = xb.to(self.device)
                    yb = yb.to(self.device)
                    logits = model(xb)
                    train_ce += F.cross_entropy(logits, yb, reduction="sum").item()
                    train_correct += int((logits.argmax(dim=1) == yb).sum().item())
                    train_total += int(yb.shape[0])
            train_acc = train_correct / max(train_total, 1)
            train_loss_ce = train_ce / max(train_total, 1)
            val_acc = float("nan")
            val_loss_ce = float("nan")
            if has_val:
                x_v = torch.tensor(X_val, dtype=torch.float32, device=self.device)
                y_v = torch.tensor(y_val, dtype=torch.long, device=self.device)
                v_logits = model(x_v)
                val_loss_ce = float(F.cross_entropy(v_logits, y_v).item())
                val_acc = float((v_logits.argmax(dim=1) == y_v).float().mean().item())
            return train_acc, train_loss_ce, val_acc, val_loss_ce

        def _run_epoch(epoch_label: str, *, rules_focus: bool) -> float:
            model.train()
            running_loss = 0.0
            for xb, yb in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                y_target = sample_alternative_targets(yb, num_classes)

                pack = model.forward_pack(xb)
                ce = F.cross_entropy(pack.logits, yb)
                ce_rules = F.cross_entropy(pack.logits_rules, yb)
                if cfg.distill_lambda > 0.0:
                    t = cfg.distill_temperature
                    kd = F.kl_div(
                        F.log_softmax(pack.logits_rules / t, dim=1),
                        F.softmax(pack.logits_hyper.detach() / t, dim=1),
                        reduction="batchmean",
                    ) * (t * t)
                else:
                    kd = torch.tensor(0.0, device=self.device)

                w_rule, _, _, _ = pack.rule_params
                sparse = w_rule.abs().mean()

                x_cf = model.generate_counterfactual(xb, y_target)
                logits_cf = model(x_cf)
                ce_cf = F.cross_entropy(logits_cf, y_target)

                x_cont = bipolar_to_continuous(xb)
                delta = x_cf - x_cont
                l1 = delta.abs().mean()
                l2 = (delta**2).mean()

                if rules_focus:
                    loss = (
                        0.2 * ce
                        + 0.8 * ce_rules
                        + cfg.distill_lambda * kd
                        + cfg.rule_sparsity_lambda * sparse
                        + 0.5 * cfg.cf_lambda * ce_cf
                        + cfg.l1_lambda * l1
                        + cfg.l2_lambda * l2
                    )
                else:
                    loss = (
                        ce
                        + 0.2 * ce_rules
                        + cfg.rule_sparsity_lambda * sparse
                        + cfg.cf_lambda * ce_cf
                        + cfg.l1_lambda * l1
                        + cfg.l2_lambda * l2
                    )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * xb.shape[0]
            avg_loss = running_loss / len(train_ds)
            if verbose:
                print(f"[{epoch_label}] loss={avg_loss:.4f}", flush=True)
            return float(avg_loss)

        # Phase 1: entrainement standard hybride
        for epoch in range(1, cfg.epochs + 1):
            avg_loss = _run_epoch(f"P1 {epoch:03d}/{cfg.epochs}", rules_focus=False)
            train_acc, train_loss_ce, val_acc, val_loss_ce = _epoch_metrics()

            if has_val and val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

            self.history.append(
                {
                    "epoch": epoch,
                    "phase": "p1",
                    "train_loss": float(avg_loss),
                    "train_loss_ce": float(train_loss_ce),
                    "val_loss": float(val_loss_ce) if has_val else float("nan"),
                    "train_accuracy": float(train_acc),
                    "val_accuracy": float(val_acc) if has_val else float("nan"),
                    "best_val_accuracy": float(best_val_acc) if has_val else float("nan"),
                }
            )

            if verbose and has_val:
                print(
                    f"[P1 val {epoch:03d}/{cfg.epochs}] train_acc={train_acc:.4f} val_acc={val_acc:.4f} "
                    f"best_val_acc={best_val_acc:.4f}",
                    flush=True,
                )

        # Phase 2: distillation hypernet -> regles
        if cfg.rules_phase_epochs > 0:
            for p in model.encoder.parameters():
                p.requires_grad = False
            for p in model.hyper.parameters():
                p.requires_grad = False
            trainable = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(
                trainable, lr=cfg.lr * cfg.rules_phase_lr_scale, weight_decay=cfg.weight_decay,
            )
            if verbose:
                print(
                    f"[Phase 2] distillation active: epochs={cfg.rules_phase_epochs}, "
                    f"lambda={cfg.distill_lambda}, T={cfg.distill_temperature}",
                    flush=True,
                )
            for epoch in range(1, cfg.rules_phase_epochs + 1):
                avg_loss = _run_epoch(f"P2 {epoch:03d}/{cfg.rules_phase_epochs}", rules_focus=True)
                train_acc, train_loss_ce, val_acc, val_loss_ce = _epoch_metrics()
                if has_val and val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                self.history.append(
                    {
                        "epoch": epoch,
                        "phase": "p2",
                        "train_loss": float(avg_loss),
                        "train_loss_ce": float(train_loss_ce),
                        "val_loss": float(val_loss_ce) if has_val else float("nan"),
                        "train_accuracy": float(train_acc),
                        "val_accuracy": float(val_acc) if has_val else float("nan"),
                        "best_val_accuracy": float(best_val_acc) if has_val else float("nan"),
                    }
                )
                if verbose and has_val:
                    print(
                        f"[P2 val {epoch:03d}/{cfg.rules_phase_epochs}] train_acc={train_acc:.4f} "
                        f"val_acc={val_acc:.4f} best_val_acc={best_val_acc:.4f}",
                        flush=True,
                    )

        if has_val and best_state is not None:
            model.load_state_dict(best_state)
            model.to(self.device)

        from hyconex_from_scratch.trainer import TrainingResult

        return TrainingResult(
            model=model,
            num_classes=num_classes,
            device=self.device,
            history=self.history,
            best_val_accuracy=float(best_val_acc) if has_val else float("nan"),
        )

    def evaluate_counterfactuals(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        max_samples: int = 4000,
    ) -> dict:
        if self.model is None or self._num_classes is None:
            raise RuntimeError("Aucun modele entraine.")

        model = self.model
        num_classes = self._num_classes
        model.eval()
        x = self._to_bipolar_np(x)
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
        x_cont = bipolar_to_continuous(x_sub)
        l1 = torch.norm(x_cf - x_cont, p=1, dim=1).mean().item()
        x_cf_bin = continuous_to_bipolar(x_cf)
        x_sub_bin = continuous_to_bipolar(x_cont)
        changed = (x_cf_bin != x_sub_bin).float().sum(dim=1).mean().item()
        return {
            "validity_cf": float(validity),
            "changed_bits_mean": float(changed),
            "hamming_mean": float(changed),
            "proximity_l1_mean": float(l1),
            "n_evaluated": int(x_sub.shape[0]),
            "input_space": "bipolar",
        }

    def export_rules(
        self,
        *,
        top_per_rule: int = 4,
        min_abs_weight: float = 0.001,
    ) -> list[dict]:
        assert self.model is not None
        model: HyConExBipolarRulesModel = self.model
        theta = model.theta_bias
        w_rule, _, w_out, _ = unpack_main_params(
            theta.unsqueeze(0),
            input_dim=model.dr_input_dim,
            num_rules=model.num_rules,
            num_classes=model.num_classes,
        )
        class_names = self.class_names or [str(i) for i in range(model.num_classes)]
        feat_names = self.rule_feature_names or [f"oh_{i}" for i in range(model.dr_input_dim)]
        return extract_rules(
            w_rule,
            w_out,
            feat_names,
            class_names,
            top_per_rule=top_per_rule,
            min_abs_weight=min_abs_weight,
        )
