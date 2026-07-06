"""
Evaluation harmonisee des contrefactuels DLBAC (memes protocoles pour tous les modeles).

Protocoles :
  A) random_target  — 1 classe alternative aleatoire par echantillon
  B) all_targets    — toutes les classes != y_true
  C) per_target     — validite moyenne par classe cible (protocole B)

Usage :
    python eval_cf_harmonized_dlbac.py
    python eval_cf_harmonized_dlbac.py --max-samples 1500
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from hyconex_pure_bipolar import BipolarRulesConfig, HyConExBipolarRulesTrainer
from hyconex_pure_bipolar.bipolar import bipolar_to_continuous, continuous_to_bipolar
from nouveau_module import HybridDRTrainer
from prepare_dlbac_datasets import discover_dlbac_datasets
from tabresnet_dlbac import TabResNetDLBACConfig, TabResNetDLBACTrainer
from train_nouveau_module_dlbac_quantile import build_onehot_splits

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "results" / "cf_eval_harmonized"


@dataclass
class CFBackend:
    label: str
    mode: str
    device: torch.device
    num_classes: int
    to_tensor: Callable[[np.ndarray], torch.Tensor]
    predict_logits: Callable[[torch.Tensor], torch.Tensor]
    generate_cf: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    proximity: Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]
    generate_all_classes: Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]] | None = None


def _subsample_idx(n: int, max_samples: int, seed: int) -> np.ndarray:
    if n <= max_samples:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=max_samples, replace=False))


def load_tabresnet(spec_name: str, splits, device: torch.device) -> CFBackend:
    res_dir = ROOT / "results" / "tabresnet_dlbac"
    cfg = TabResNetDLBACConfig(**json.loads((res_dir / f"{spec_name}_config.json").read_text(encoding="utf-8")))
    ckpt = torch.load(res_dir / f"{spec_name}_model.pt", map_location=device, weights_only=False)
    mode = ckpt["mode"]

    if mode == "instance":
        trainer = TabResNetDLBACTrainer(cfg, device=str(device))
        trainer._hybrid = HybridDRTrainer(trainer._hybrid_config(), device=str(device))
        trainer._hybrid.binarizer.fit_transform(splits.x_train, feature_names=splits.feature_names)
        dim = trainer._hybrid.binarizer.transform(splits.x_train[:1]).shape[1]
        nc = splits.num_classes
        trainer._hybrid._build_model(dim, nc)
        trainer._hybrid.model.load_state_dict(ckpt["state_dict"])
        trainer._hybrid.model.eval()
        model = trainer._hybrid.model
        binarizer = trainer._hybrid.binarizer

        def to_tensor(x_np: np.ndarray) -> torch.Tensor:
            z = binarizer.transform(np.asarray(x_np, dtype=np.float32))
            return torch.tensor(z, dtype=torch.float32, device=device)

        def proximity(x_t: torch.Tensor, x_cf: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            flips = (x_cf != x_t).float().sum(dim=1)
            x0 = torch.tensor(binarizer.binary_to_continuous(x_t.cpu().numpy()), device=device, dtype=torch.float32)
            x1 = torch.tensor(binarizer.binary_to_continuous(x_cf.cpu().numpy()), device=device, dtype=torch.float32)
            return flips, (x1 - x0).abs().sum(dim=1)

        return CFBackend(
            label="tabresnet_dlbac",
            mode="instance",
            device=device,
            num_classes=nc,
            to_tensor=to_tensor,
            predict_logits=model.predict_logits,
            generate_cf=lambda x, y: model.generate_counterfactual_binary(x, y)[0],
            proximity=proximity,
            generate_all_classes=model.generate_counterfactuals_all_classes,
        )

    if mode == "bipolar_hyper":
        trainer = TabResNetDLBACTrainer(cfg, device=str(device))
        bcfg = trainer._bipolar_config(splits.x_train.shape[1], splits.num_classes)
        bip = HyConExBipolarRulesTrainer(bcfg, device=str(device))
        bip._ensure_model(splits.x_train.shape[1], splits.num_classes)
        bip.model.load_state_dict(ckpt["state_dict"])
        bip.model.eval()
        model = bip.model
        nc = splits.num_classes

        def to_tensor(x_np: np.ndarray) -> torch.Tensor:
            z = continuous_to_bipolar(np.asarray(x_np, dtype=np.float32))
            return torch.tensor(z, dtype=torch.float32, device=device)

        def generate_cf(x_t: torch.Tensor, y_tgt: torch.Tensor) -> torch.Tensor:
            return model.generate_counterfactual(x_t, y_tgt)

        def predict_logits(x_in: torch.Tensor) -> torch.Tensor:
            return model(x_in)

        def proximity(x_t: torch.Tensor, x_cf: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            x_cont = bipolar_to_continuous(x_t)
            if x_cf.shape[-1] == x_t.shape[-1]:
                x_cf_cont = x_cf if x_cf.max() <= 1.01 else bipolar_to_continuous(x_cf)
            else:
                x_cf_cont = x_cf
            l1 = (x_cf_cont - x_cont).abs().sum(dim=1)
            flips = (continuous_to_bipolar(x_cf_cont) != continuous_to_bipolar(x_cont)).float().sum(dim=1)
            return flips, l1

        return CFBackend(
            label="tabresnet_dlbac",
            mode="bipolar_hyper",
            device=device,
            num_classes=nc,
            to_tensor=to_tensor,
            predict_logits=predict_logits,
            generate_cf=generate_cf,
            proximity=proximity,
            generate_all_classes=None,
        )

    raise ValueError(f"Mode checkpoint inconnu: {mode}")


def load_hyconex_bipolar(spec_name: str, device: torch.device) -> CFBackend:
    ckpt = torch.load(
        ROOT / "results" / "hyconex_pure_bipolar_dlbac" / f"{spec_name}_model.pt",
        map_location=device,
        weights_only=False,
    )
    cfg = BipolarRulesConfig(**ckpt["config"])
    trainer = HyConExBipolarRulesTrainer(cfg, device=str(device))
    trainer._ensure_model(int(ckpt["input_dim"]), int(ckpt["num_classes"]))
    trainer.model.load_state_dict(ckpt["state_dict"])
    trainer.model.eval()
    model = trainer.model
    nc = int(ckpt["num_classes"])

    def to_tensor(x_np: np.ndarray) -> torch.Tensor:
        z = continuous_to_bipolar(np.asarray(x_np, dtype=np.float32))
        return torch.tensor(z, dtype=torch.float32, device=device)

    def generate_cf(x_t: torch.Tensor, y_tgt: torch.Tensor) -> torch.Tensor:
        return model.generate_counterfactual(x_t, y_tgt)

    def proximity(x_t: torch.Tensor, x_cf: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_cont = bipolar_to_continuous(x_t)
        x_cf_cont = x_cf if x_cf.max() <= 1.01 + 1e-3 else bipolar_to_continuous(x_cf)
        l1 = (x_cf_cont - x_cont).abs().sum(dim=1)
        flips = (continuous_to_bipolar(x_cf_cont) != continuous_to_bipolar(x_cont)).float().sum(dim=1)
        return flips, l1

    return CFBackend(
        label="hyconex_pure_bipolar",
        mode="bipolar",
        device=device,
        num_classes=nc,
        to_tensor=to_tensor,
        predict_logits=model.forward,
        generate_cf=generate_cf,
        proximity=proximity,
        generate_all_classes=None,
    )


@torch.no_grad()
def eval_random_target(
    backend: CFBackend,
    x: np.ndarray,
    y: np.ndarray,
    *,
    max_samples: int,
    seed: int,
    batch_size: int = 64,
) -> dict[str, Any]:
    idx = _subsample_idx(len(y), max_samples, seed)
    nc = backend.num_classes
    rng = np.random.default_rng(seed + 7)
    targets = np.array(
        [int(rng.choice([c for c in range(nc) if c != int(yi)])) for yi in y[idx]],
        dtype=np.int64,
    )

    valid, flips_all, l1_all = [], [], []
    for start in range(0, len(idx), batch_size):
        sl = idx[start : start + batch_size]
        tg = targets[start : start + batch_size]
        x_t = backend.to_tensor(x[sl])
        y_tgt = torch.tensor(tg, dtype=torch.long, device=backend.device)
        x_cf = backend.generate_cf(x_t, y_tgt)
        logits = backend.predict_logits(x_cf if x_cf.shape == x_t.shape else x_cf)
        valid.extend((logits.argmax(dim=1) == y_tgt).float().cpu().tolist())
        fl, l1 = backend.proximity(x_t, x_cf)
        flips_all.extend(fl.cpu().tolist())
        l1_all.extend(l1.cpu().tolist())

    return {
        "protocol": "random_target",
        "n_evaluated": int(len(idx)),
        "validity": float(np.mean(valid)),
        "changed_bits_mean": float(np.mean(flips_all)),
        "proximity_l1_mean": float(np.mean(l1_all)),
    }


@torch.no_grad()
def eval_all_targets(
    backend: CFBackend,
    x: np.ndarray,
    y: np.ndarray,
    *,
    max_samples: int,
    seed: int,
    batch_size: int = 32,
) -> dict[str, Any]:
    idx = _subsample_idx(len(y), max_samples, seed)
    nc = backend.num_classes
    per_target: dict[int, list[float]] = {c: [] for c in range(nc)}
    flip_pairs: list[float] = []
    l1_pairs: list[float] = []
    valid_pairs: list[float] = []

    if backend.generate_all_classes is not None:
        y_sub = y[idx]
        x_t_all = backend.to_tensor(x[idx])
        y_t = torch.tensor(y_sub, dtype=torch.long, device=backend.device)
        for start in range(0, x_t_all.shape[0], batch_size):
            x_t = x_t_all[start : start + batch_size]
            y_b = y_t[start : start + batch_size]
            x_cf_all, logits_cf_all = backend.generate_all_classes(x_t, y_b)
            class_ids = torch.arange(nc, device=backend.device).view(1, nc).expand(x_t.shape[0], -1)
            valid_mask = class_ids != y_b.unsqueeze(1)
            preds = logits_cf_all.argmax(dim=2)
            hits = (preds == class_ids).float()
            for b in range(x_t.shape[0]):
                for c in range(nc):
                    if int(y_b[b].item()) == c:
                        continue
                    ok = float(hits[b, c].item())
                    valid_pairs.append(ok)
                    per_target[c].append(ok)
            flips_mat = (x_cf_all != x_t.unsqueeze(1)).float().sum(dim=2)
            for b in range(x_t.shape[0]):
                for c in range(nc):
                    if int(y_b[b].item()) == c:
                        continue
                    flip_pairs.append(float(flips_mat[b, c].item()))
                    _, l1v = backend.proximity(x_t[b : b + 1], x_cf_all[b : b + 1, c])
                    l1_pairs.append(float(l1v.item()))
    else:
        for i in idx:
            x_t = backend.to_tensor(x[i : i + 1])
            y_true = int(y[i])
            for tgt in range(nc):
                if tgt == y_true:
                    continue
                y_tgt = torch.tensor([tgt], dtype=torch.long, device=backend.device)
                x_cf = backend.generate_cf(x_t, y_tgt)
                logits = backend.predict_logits(x_cf)
                ok = float(int(logits.argmax(dim=1).item() == tgt))
                valid_pairs.append(ok)
                per_target[tgt].append(ok)
                fl, l1 = backend.proximity(x_t, x_cf)
                flip_pairs.append(float(fl.item()))
                l1_pairs.append(float(l1.item()))

    per_target_summary = {
        str(c): float(np.mean(v)) for c, v in per_target.items() if v
    }
    return {
        "protocol": "all_targets",
        "n_evaluated_samples": int(len(idx)),
        "n_pairs": int(len(valid_pairs)),
        "validity": float(np.mean(valid_pairs)) if valid_pairs else float("nan"),
        "changed_bits_mean": float(np.mean(flip_pairs)) if flip_pairs else float("nan"),
        "proximity_l1_mean": float(np.mean(l1_pairs)) if l1_pairs else float("nan"),
        "validity_per_target": per_target_summary,
    }


def evaluate_backend(
    backend: CFBackend,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    max_samples: int,
    seed: int,
) -> dict[str, Any]:
    random_m = eval_random_target(backend, x_test, y_test, max_samples=max_samples, seed=seed)
    all_m = eval_all_targets(backend, x_test, y_test, max_samples=max_samples, seed=seed)
    return {
        "model": backend.label,
        "mode": backend.mode,
        "protocols": {
            "random_target": random_m,
            "all_targets": all_m,
        },
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluation CF harmonisee DLBAC")
    p.add_argument("--dataset", nargs="*", default=["u4k-r4k-auth11k", "amazon1"])
    p.add_argument("--max-samples", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    specs = {s.name: s for s in discover_dlbac_datasets() if s.has_train}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []

    for name in args.dataset:
        if name not in specs:
            raise SystemExit(f"Jeu introuvable: {name}")
        splits = build_onehot_splits(specs[name], val_size=0.2, random_state=42, use_cache=True)
        print(f"\n=== {name} (max_samples={args.max_samples}) ===", flush=True)

        backends: list[CFBackend] = []
        try:
            backends.append(load_tabresnet(name, splits, device))
        except Exception as exc:  # noqa: BLE001
            print(f"  [skip] tabresnet_dlbac: {exc}", flush=True)
        try:
            backends.append(load_hyconex_bipolar(name, device))
        except Exception as exc:  # noqa: BLE001
            print(f"  [skip] hyconex_pure_bipolar: {exc}", flush=True)

        for backend in backends:
            row = evaluate_backend(
                backend,
                splits.x_test,
                splits.y_test,
                max_samples=args.max_samples,
                seed=args.seed,
            )
            row["dataset"] = name
            all_rows.append(row)
            rnd = row["protocols"]["random_target"]
            allt = row["protocols"]["all_targets"]
            print(
                f"  {backend.label} ({backend.mode})",
                flush=True,
            )
            print(
                f"    random_target : valid={rnd['validity']:.4f} "
                f"flips={rnd['changed_bits_mean']:.2f} l1={rnd['proximity_l1_mean']:.2f}",
                flush=True,
            )
            print(
                f"    all_targets   : valid={allt['validity']:.4f} "
                f"flips={allt['changed_bits_mean']:.2f} l1={allt['proximity_l1_mean']:.2f} "
                f"pairs={allt['n_pairs']}",
                flush=True,
            )

        (OUT_DIR / f"{name}_cf_eval.json").write_text(
            json.dumps([r for r in all_rows if r["dataset"] == name], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    (OUT_DIR / "summary.json").write_text(
        json.dumps(all_rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nResultats -> {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
