from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from hyconex_from_scratch.trainer import HyConExTrainer, sample_alternative_targets, set_seed
from hyconex_pure_rules.config import RulesConfig
from hyconex_pure_rules.model import HyConExLocalRulesModel
from nouveau_module.main_rule_net import extract_rules, unpack_main_params


class HyConExLocalRulesTrainer(HyConExTrainer):
    def __init__(self, config: RulesConfig | None = None, *, device=None) -> None:
        super().__init__(config=config or RulesConfig(), device=device)
        self.config: RulesConfig = self.config  # type: ignore[assignment]
        self.feature_names: list[str] = []

    def _ensure_model(self, input_dim: int, num_classes: int) -> HyConExLocalRulesModel:
        cfg: RulesConfig = self.config
        self._num_classes = num_classes
        self.model = HyConExLocalRulesModel(
            input_dim=input_dim,
            num_classes=num_classes,
            latent_dim=cfg.latent_dim,
            hidden_dim=cfg.hidden_dim,
            num_rules=cfg.num_rules,
            temperature=cfg.temperature,
            hyper_weight=cfg.hyper_weight,
            rule_weight=cfg.rule_weight,
            ctx_modulation=cfg.ctx_modulation,
        ).to(self.device)
        return self.model

    def fit(self, X_train, y_train, X_val=None, y_val=None, *, verbose: bool = True):
        cfg: RulesConfig = self.config
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
        best_state = None
        self.history = []

        for epoch in range(1, cfg.epochs + 1):
            model.train()
            running_loss = 0.0
            for xb, yb in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                y_target = sample_alternative_targets(yb, num_classes)

                pack = model.forward_pack(xb)
                ce = F.cross_entropy(pack.logits, yb)

                w_rule, _, _, _ = pack.rule_params
                sparse = w_rule.abs().mean()

                x_cf = model.generate_counterfactual(xb, y_target)
                logits_cf = model(x_cf)
                ce_cf = F.cross_entropy(logits_cf, y_target)
                delta = x_cf - xb
                l1 = delta.abs().mean()
                l2 = (delta**2).mean()

                loss = (
                    ce
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
            val_acc = float("nan")
            if has_val:
                val_metrics = self._evaluate_arrays(model, X_val, y_val)  # type: ignore[arg-type]
                val_acc = float(val_metrics["accuracy"])

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

        from hyconex_from_scratch.trainer import TrainingResult

        return TrainingResult(
            model=model,
            num_classes=num_classes,
            device=self.device,
            history=self.history,
            best_val_accuracy=float(best_val_acc) if has_val else float("nan"),
        )

    def export_rules(
        self,
        *,
        top_per_rule: int = 4,
        min_abs_weight: float = 0.001,
    ) -> list[dict]:
        assert self.model is not None
        model: HyConExLocalRulesModel = self.model
        feat_names = [f"z_{i}" for i in range(model.latent_dim)]
        theta = model.theta_bias
        w_rule, _, w_out, _ = unpack_main_params(
            theta.unsqueeze(0),
            input_dim=model.latent_dim,
            num_rules=model.num_rules,
            num_classes=model.num_classes,
        )
        class_names = getattr(self, "class_names", None) or [str(i) for i in range(model.num_classes)]
        return extract_rules(
            w_rule,
            w_out,
            feat_names,
            class_names,
            top_per_rule=top_per_rule,
            min_abs_weight=min_abs_weight,
        )
