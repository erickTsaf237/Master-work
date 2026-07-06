from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score

from hyconex_pure_bipolar import BipolarRulesConfig, HyConExBipolarRulesTrainer
from hyconex_pure_bipolar.bipolar import continuous_to_bipolar
from hyperlogic_pure import PureDRConfig, PureDRNetTrainer
from nouveau_module import HybridDRConfig, HybridDRTrainer, TrainingResult
from tabresnet_dlbac.config import TabResNetDLBACConfig


@dataclass
class TabResNetDLBACResult:
    mode: Literal["instance", "embed", "bipolar_hyper"]
    best_val_accuracy: float
    best_val_auroc: float
    history: list[dict]


class TabResNetDLBACTrainer:
    """
    Façade DLBAC performante :
    - dim <= 512 : TabResNet instance (nouveau_module / Dry Bean)
    - dim > 512  : HyConEx bipolar + TabResNet hypernet (Amazon, éprouvé)
    """

    def __init__(self, config: TabResNetDLBACConfig | None = None, *, device: str | None = "auto") -> None:
        self.config = config or TabResNetDLBACConfig()
        self._device_arg = device
        self.mode: Literal["instance", "embed", "bipolar_hyper"] | None = None
        self._hybrid: HybridDRTrainer | None = None
        self._pure: PureDRNetTrainer | None = None
        self._bipolar: HyConExBipolarRulesTrainer | None = None
        self.class_names: list[str] = []
        self.feature_names: list[str] = []
        self._fallback_device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device in (None, "auto")
            else torch.device(device)
        )

    @property
    def device(self) -> torch.device:
        if self._hybrid is not None:
            return self._hybrid.device
        if self._pure is not None:
            return self._pure.device
        if self._bipolar is not None:
            return self._bipolar.device
        return self._fallback_device

    @property
    def binarizer(self):
        if self._hybrid is not None:
            return self._hybrid.binarizer
        raise AttributeError("binarizer disponible uniquement en mode instance")

    @property
    def model(self):
        if self.mode == "instance" and self._hybrid is not None:
            return self._hybrid.model
        if self.mode == "embed" and self._pure is not None:
            return self._pure.model
        if self.mode == "bipolar_hyper" and self._bipolar is not None:
            return self._bipolar.model
        return None

    def _bipolar_config(self, n_features: int, num_classes: int) -> BipolarRulesConfig:
        c = self.config
        high = n_features > 1000
        return BipolarRulesConfig(
            seed=c.seed,
            epochs=c.embed_epochs,
            batch_size=c.embed_batch_size if high else 64,
            lr=8e-4 if high else 1e-3,
            latent_dim=64 if high else 32,
            hidden_dim=c.hyper_hidden_dim,
            cf_lambda=0.25 if high else 0.35,
            num_rules=c.embed_num_rules,
            temperature=0.5 if high else 0.6,
            hyper_weight=0.78,
            rule_weight=0.22,
            flip_lambda=c.embed_flip_lambda,
            distill_lambda=1.6 if high else 1.2,
            rules_phase_epochs=6 if high else 10,
        )

    def _hybrid_config(self) -> HybridDRConfig:
        c = self.config
        return HybridDRConfig(
            seed=c.seed,
            epochs=c.instance_epochs,
            batch_size=c.instance_batch_size,
            lr=c.instance_lr,
            num_rules=c.instance_num_rules,
            hyper_hidden_dim=c.hyper_hidden_dim,
            cf_hidden_dim=c.cf_hidden_dim,
            tabresnet_n_blocks=c.tabresnet_n_blocks,
            tabresnet_dropout=c.tabresnet_dropout,
            temperature=c.instance_temperature,
            bins_per_feature=4,
            input_encoding=c.input_encoding,
            use_class_weights=c.use_class_weights,
            early_stop_metric=c.early_stop_metric,
            cf_warmup_epochs=c.instance_cf_warmup,
            cf_lambda=c.instance_cf_lambda,
            flip_lambda=c.instance_flip_lambda,
            rule_sparsity_lambda=c.rule_sparsity_lambda,
        )

    def _pure_config(self, *, distill: bool) -> PureDRConfig:
        c = self.config
        return PureDRConfig(
            seed=c.seed,
            epochs=c.embed_epochs,
            batch_size=c.embed_batch_size,
            lr=c.embed_lr,
            num_rules=c.embed_num_rules,
            temperature=c.embed_temperature,
            hyper_hidden_dim=c.hyper_hidden_dim,
            cf_hidden_dim=c.cf_hidden_dim,
            tabresnet_n_blocks=c.tabresnet_n_blocks,
            tabresnet_dropout=c.tabresnet_dropout,
            embed_dim_high=c.embed_dim,
            early_stop_metric=c.early_stop_metric,
            max_instance_dim=c.max_instance_dim,
            cf_lambda=0.0,
            flip_lambda=0.0,
            cf_epochs=c.embed_cf_epochs,
            cf_lambda_phase2=c.embed_cf_lambda,
            flip_lambda_phase2=c.embed_flip_lambda,
            use_class_weights=not distill,
            distill_lambda=c.embed_distill_lambda if distill else 0.0,
            distill_only=False,
        )

    @staticmethod
    def _fit_svm_teacher(x_train: np.ndarray, y_train: np.ndarray, *, seed: int) -> np.ndarray:
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.svm import LinearSVC

        clf = CalibratedClassifierCV(
            LinearSVC(class_weight="balanced", max_iter=3000, random_state=seed),
            cv=3,
            method="sigmoid",
        )
        clf.fit(np.asarray(x_train, dtype=np.float32), y_train)
        return clf.predict_proba(x_train).astype(np.float32)

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
    ) -> TabResNetDLBACResult:
        x_train = np.asarray(x_train, dtype=np.float32)
        self.feature_names = feature_names or [f"oh_{i}" for i in range(x_train.shape[1])]
        self.class_names = class_names or []

        if x_train.shape[1] <= self.config.max_instance_dim:
            return self._fit_instance(
                x_train, y_train, x_val, y_val, feature_names=feature_names, class_names=class_names, verbose=verbose
            )
        return self._fit_bipolar_hyper(
            x_train, y_train, x_val, y_val, feature_names=feature_names, class_names=class_names, verbose=verbose
        )

    def _fit_instance(self, x_train, y_train, x_val, y_val, *, feature_names, class_names, verbose) -> TabResNetDLBACResult:
        self.mode = "instance"
        self._hybrid = HybridDRTrainer(self._hybrid_config(), device=self._device_arg)
        if verbose:
            print("  Strategie: TabResNet instance (HybridDRNetModel)", flush=True)
        result: TrainingResult = self._hybrid.fit(
            x_train,
            y_train,
            x_val_cont=x_val,
            y_val=y_val,
            feature_names=feature_names,
            class_names=class_names,
            verbose=verbose,
        )
        self.class_names = result.class_names
        return TabResNetDLBACResult(
            mode="instance",
            best_val_accuracy=float(result.best_val_accuracy),
            best_val_auroc=float(result.best_val_auroc),
            history=result.history,
        )

    def _fit_bipolar_hyper(self, x_train, y_train, x_val, y_val, *, feature_names, class_names, verbose) -> TabResNetDLBACResult:
        self.mode = "bipolar_hyper"
        cfg = self._bipolar_config(x_train.shape[1], int(np.max(y_train) + 1))
        self._bipolar = HyConExBipolarRulesTrainer(cfg, device=self._device_arg)
        self._bipolar.feature_names = feature_names or self.feature_names
        self._bipolar.class_names = class_names or self.class_names
        self.class_names = class_names or self.class_names
        if verbose:
            print("  Strategie: TabResNet hypernet + regles bipolar (Amazon / haute dim)", flush=True)
        result = self._bipolar.fit(
            x_train,
            y_train,
            X_val=x_val,
            y_val=y_val,
            verbose=verbose,
        )
        best_auroc = float("nan")
        if x_val is not None and y_val is not None:
            m = self._evaluate_bipolar(x_val, y_val, counterfactuals=False)
            best_auroc = float(m.get("auroc_ovr") or float("nan"))
        return TabResNetDLBACResult(
            mode="bipolar_hyper",
            best_val_accuracy=float(result.best_val_accuracy),
            best_val_auroc=best_auroc,
            history=result.history,
        )

    def _evaluate_bipolar(self, x: np.ndarray, y: np.ndarray, *, counterfactuals: bool) -> dict[str, Any]:
        assert self._bipolar is not None and self._bipolar.model is not None
        x_bin = np.asarray(continuous_to_bipolar(x), dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        self._bipolar.model.eval()
        probs: list[np.ndarray] = []
        bs = 256
        with torch.no_grad():
            for i in range(0, x_bin.shape[0], bs):
                xb = torch.tensor(x_bin[i : i + bs], dtype=torch.float32, device=self.device)
                pb = torch.softmax(self._bipolar.model(xb), dim=1).cpu().numpy()
                probs.append(pb)
        proba = np.vstack(probs)
        pred = np.argmax(proba, axis=1)
        out: dict[str, Any] = {"accuracy": float(accuracy_score(y, pred))}
        if proba.shape[1] == 2:
            out["auroc_ovr"] = float(roc_auc_score(y, proba[:, 1]))
        else:
            out["auroc_ovr"] = float(roc_auc_score(y, proba, multi_class="ovr"))
        if counterfactuals:
            out["counterfactuals"] = self._bipolar.evaluate_counterfactuals(x, y)
        return out

    def _bipolar_predict_proba(self, x: np.ndarray) -> np.ndarray:
        assert self._bipolar is not None and self._bipolar.model is not None
        x_bin = np.asarray(continuous_to_bipolar(x), dtype=np.float32)
        self._bipolar.model.eval()
        probs: list[np.ndarray] = []
        bs = 256
        with torch.no_grad():
            for i in range(0, x_bin.shape[0], bs):
                xb = torch.tensor(x_bin[i : i + bs], dtype=torch.float32, device=self.device)
                probs.append(torch.softmax(self._bipolar.model(xb), dim=1).cpu().numpy())
        return np.vstack(probs)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.mode == "bipolar_hyper" and self._bipolar is not None:
            return self._bipolar_predict_proba(x)
        if self.mode == "instance" and self._hybrid is not None:
            return self._hybrid.predict_proba(x)
        if self.mode == "embed" and self._pure is not None:
            return self._pure.predict_proba(x)
        raise RuntimeError("Modele non entraine.")

    def evaluate(self, x: np.ndarray, y: np.ndarray, *, counterfactuals: bool = True) -> dict[str, Any]:
        if self.mode == "bipolar_hyper" and self._bipolar is not None:
            return self._evaluate_bipolar(x, y, counterfactuals=counterfactuals)
        if self.mode == "instance" and self._hybrid is not None:
            return self._hybrid.evaluate(x, y, counterfactuals=counterfactuals)
        if self.mode == "embed" and self._pure is not None:
            return self._pure.evaluate(x, y, counterfactuals=counterfactuals)
        raise RuntimeError("Modele non entraine.")

    def export_rules(self, *, top_per_rule: int = 4, min_abs_weight: float = 0.001) -> list[dict]:
        if self.mode == "bipolar_hyper" and self._bipolar is not None:
            return self._bipolar.export_rules(top_per_rule=top_per_rule, min_abs_weight=min_abs_weight)
        if self.mode == "instance" and self._hybrid is not None:
            return self._hybrid.export_rules(top_per_rule=top_per_rule, min_abs_weight=min_abs_weight)
        if self.mode == "embed" and self._pure is not None:
            return self._pure.export_rules(top_per_rule=top_per_rule, min_abs_weight=min_abs_weight)
        raise RuntimeError("Modele non entraine.")

    def save_checkpoint(self, path: Path | str) -> None:
        if self.mode == "bipolar_hyper" and self._bipolar is not None and self._bipolar.model is not None:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "state_dict": self._bipolar.model.state_dict(),
                    "class_names": self.class_names,
                    "feature_names": self.feature_names,
                    "mode": "bipolar_hyper",
                },
                path,
            )
            return
        if self.mode == "embed" and self._pure is not None:
            self._pure.save_checkpoint(path)
            return
        if self.mode == "instance" and self._hybrid is not None and self._hybrid.model is not None:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "state_dict": self._hybrid.model.state_dict(),
                    "binarizer_mode": self._hybrid.binarizer.mode_,
                    "class_names": self.class_names,
                    "feature_names": self.feature_names,
                    "mode": "instance",
                },
                path,
            )
            return
        raise RuntimeError("Rien a sauvegarder.")
