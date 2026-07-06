"""
HyConEx PUR sur DLBAC (pipeline DLBACα + encodeur/hypernet/CF du papier).

- Prédiction : hyperréseau dynamique W(z)·z + b(z)
- Explication : contrefactuels HyConEx + attributs par gradient (top features oh_*)

Usage :
    python train_hyconex_pure_dlbac.py --dataset u4k-r4k-auth11k amazon1 --save
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from hyconex_from_scratch import HyConExTrainer, TrainConfig
from prepare_dlbac_datasets import discover_dlbac_datasets
from train_nouveau_module_dlbac_quantile import build_onehot_splits

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results" / "hyconex_pure_dlbac"


def discover_specs():
    return [s for s in discover_dlbac_datasets() if s.has_train]


def config_for_dataset(name: str, n_features: int, num_classes: int) -> TrainConfig:
    is_amazon = name.startswith("amazon")
    high = n_features > 1000
    return TrainConfig(
        seed=42,
        epochs=25 if is_amazon else 35,
        batch_size=32 if high else 128,
        lr=8e-4 if high else 1e-3,
        latent_dim=64 if high else 32,
        hidden_dim=128 if high else 64,
        cf_lambda=0.35 if not high else 0.25,
        l1_lambda=0.02 if high else 0.01,
        l2_lambda=0.005,
    )


def batched_predict_proba(
    trainer: HyConExTrainer,
    x: np.ndarray,
    *,
    batch_size: int = 256,
) -> np.ndarray:
    assert trainer.model is not None
    trainer.model.eval()
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            xb = torch.tensor(x[start : start + batch_size], dtype=torch.float32, device=trainer.device)
            proba = torch.softmax(trainer.model(xb), dim=1).cpu().numpy()
            chunks.append(proba)
    return np.vstack(chunks)


def batched_accuracy(trainer: HyConExTrainer, x: np.ndarray, y: np.ndarray, *, batch_size: int = 256) -> float:
    proba = batched_predict_proba(trainer, x, batch_size=batch_size)
    y_pred = np.argmax(proba, axis=1)
    return float((y_pred == y).mean())


def explain_counterfactual(
    trainer: HyConExTrainer,
    x: np.ndarray,
    sample_idx: int,
    target_class: int,
    *,
    feature_names: list[str],
    class_names: list[str],
    y_true: int | None = None,
    top_k: int = 12,
) -> dict:
    assert trainer.model is not None
    x_row = np.asarray(x[sample_idx : sample_idx + 1], dtype=np.float32)
    x_t = torch.tensor(x_row, device=trainer.device, requires_grad=False)
    y_tgt = torch.tensor([target_class], dtype=torch.long, device=trainer.device)

    with torch.no_grad():
        logits = trainer.model(x_t)
        proba = torch.softmax(logits, dim=1)
        y_pred = int(proba.argmax(dim=1).item())
        proba_orig = float(proba[0, y_pred].item())
        x_cf = trainer.model.generate_counterfactual(x_t, y_tgt)
        logits_cf = trainer.model(x_cf)
        proba_cf = torch.softmax(logits_cf, dim=1)
        y_cf = int(proba_cf.argmax(dim=1).item())
        proba_cf_v = float(proba_cf[0, y_cf].item())

    delta = (x_cf - x_t).detach().cpu().numpy()[0]
    x_np = x_row[0]
    changes = []
    for j in range(len(delta)):
        if abs(float(delta[j])) > 1e-4:
            changes.append(
                {
                    "feature": feature_names[j],
                    "from": float(x_np[j]),
                    "to": float(x_np[j] + delta[j]),
                    "delta": float(delta[j]),
                }
            )
    changes.sort(key=lambda c: abs(c["delta"]), reverse=True)

    return {
        "sample_idx": sample_idx,
        "y_true": y_true,
        "y_pred_orig": y_pred,
        "y_pred_orig_name": class_names[y_pred] if y_pred < len(class_names) else str(y_pred),
        "proba_orig": proba_orig,
        "y_target": target_class,
        "y_target_name": class_names[target_class] if target_class < len(class_names) else str(target_class),
        "y_pred_cf": y_cf,
        "y_pred_cf_name": class_names[y_cf] if y_cf < len(class_names) else str(y_cf),
        "proba_cf": proba_cf_v,
        "valid": y_cf == target_class,
        "n_changes": len(changes),
        "changes": changes[:top_k],
    }


def explain_gradient_saliency(
    trainer: HyConExTrainer,
    x: np.ndarray,
    sample_idx: int,
    *,
    feature_names: list[str],
    class_names: list[str],
    top_k: int = 12,
) -> dict:
    """Attribution locale : gradient de la classe prédite par rapport aux entrées."""
    assert trainer.model is not None
    x_row = np.asarray(x[sample_idx : sample_idx + 1], dtype=np.float32)
    x_t = torch.tensor(x_row, device=trainer.device, requires_grad=True)
    trainer.model.eval()
    logits = trainer.model(x_t)
    y_pred = int(logits.argmax(dim=1).item())
    score = logits[0, y_pred]
    score.backward()
    grad = x_t.grad.detach().cpu().numpy()[0]
    abs_g = np.abs(grad)
    top_idx = np.argsort(abs_g)[::-1][:top_k]
    attrs = [
        {"feature": feature_names[i], "gradient": float(grad[i]), "value": float(x_row[0, i])}
        for i in top_idx
    ]
    return {
        "sample_idx": sample_idx,
        "y_pred": y_pred,
        "y_pred_name": class_names[y_pred] if y_pred < len(class_names) else str(y_pred),
        "top_features": attrs,
    }


def pick_cf_example(trainer, x_test, y_test, class_names: list[str], max_probe: int = 32) -> tuple[int, int] | None:
    for i in range(min(max_probe, len(y_test))):
        with torch.no_grad():
            x_t = torch.tensor(x_test[i : i + 1], dtype=torch.float32, device=trainer.device)
            pred = int(trainer.model(x_t).argmax().item())
        for target in range(len(class_names)):
            if target == pred:
                continue
            cf = explain_counterfactual(
                trainer,
                x_test,
                i,
                target,
                feature_names=[f"oh_{j}" for j in range(x_test.shape[1])],
                class_names=class_names,
                y_true=int(y_test[i]),
            )
            if cf["valid"]:
                return i, target
    return None


def train_one(spec, *, save_dir: Path | None = None, verbose: bool = True) -> dict:
    splits = build_onehot_splits(spec, val_size=0.2, random_state=42, use_cache=True)
    cfg = config_for_dataset(spec.name, splits.x_train.shape[1], splits.num_classes)
    trainer = HyConExTrainer(cfg)

    if verbose:
        print(f"\n=== {spec.name} (HyConEx PUR) ===", flush=True)
        print(f"  features: {splits.x_train.shape[1]} | classes: {splits.num_classes}", flush=True)
        print(f"  Device: {trainer.device}", flush=True)

    result = trainer.fit(
        splits.x_train,
        splits.y_train,
        X_val=splits.x_val,
        y_val=splits.y_val,
        verbose=verbose,
    )

    high_dim = splits.x_train.shape[1] > 1000
    cf_samples = 512 if high_dim else 4000
    eval_bs = 64 if high_dim else 256
    metrics = trainer.evaluate(
        splits.x_test,
        splits.y_test,
        counterfactuals=True,
        cf_max_samples=cf_samples,
    )
    if high_dim:
        metrics["accuracy"] = batched_accuracy(trainer, splits.x_test, splits.y_test, batch_size=eval_bs)

    from sklearn.metrics import roc_auc_score

    proba = batched_predict_proba(trainer, splits.x_test, batch_size=eval_bs)
    if splits.num_classes == 2:
        test_auroc = float(roc_auc_score(splits.y_test, proba[:, 1]))
    else:
        test_auroc = float(metrics.get("auroc_ovr") or roc_auc_score(splits.y_test, proba, multi_class="ovr"))

    example_cf = None
    example_saliency = None
    idx_ex = 0
    picked = None
    try:
        max_probe = 8 if high_dim else 32
        picked = pick_cf_example(
            trainer, splits.x_test, splits.y_test, splits.class_names, max_probe=max_probe
        )
    except Exception:  # noqa: BLE001
        picked = None
    if picked is not None:
        idx_ex, target = picked
    else:
        idx_ex = 0
        target = 1 if splits.num_classes > 1 and int(splits.y_test[0]) == 0 else 0

    example_cf = explain_counterfactual(
        trainer,
        splits.x_test,
        idx_ex,
        target,
        feature_names=splits.feature_names,
        class_names=splits.class_names,
        y_true=int(splits.y_test[idx_ex]),
    )
    example_saliency = explain_gradient_saliency(
        trainer,
        splits.x_test,
        idx_ex,
        feature_names=splits.feature_names,
        class_names=splits.class_names,
    )

    model_path = None
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        model_path = save_dir / f"{spec.name}_model.pt"
        torch.save(
            {
                "state_dict": trainer.model.state_dict(),
                "config": asdict(cfg),
                "class_names": splits.class_names,
                "feature_names": splits.feature_names,
                "input_dim": splits.x_train.shape[1],
                "num_classes": splits.num_classes,
            },
            model_path,
        )

    is_amazon = spec.name.startswith("amazon")
    threshold = 0.72 if is_amazon else 0.88

    return {
        "dataset": spec.name,
        "model": "HyConExFromScratch",
        "pure_hyconex": True,
        "num_features": int(splits.x_train.shape[1]),
        "num_classes": splits.num_classes,
        "best_val_accuracy": float(result.best_val_accuracy),
        "test_accuracy": float(metrics["accuracy"]),
        "test_auroc": test_auroc,
        "acceptable": test_auroc >= threshold,
        "accept_threshold": threshold,
        "cf_validity": float(metrics.get("counterfactuals", {}).get("validity_cf", 0.0)),
        "cf_proximity_l1": float(metrics.get("counterfactuals", {}).get("proximity_l1_mean", 0.0)),
        "example_counterfactual": example_cf,
        "example_saliency": example_saliency,
        "model_checkpoint": str(model_path) if model_path else None,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="HyConEx pur sur DLBAC")
    p.add_argument("--dataset", nargs="*", default=["u4k-r4k-auth11k", "amazon1"])
    p.add_argument("--save", action="store_true")
    args = p.parse_args()

    specs = {s.name: s for s in discover_specs()}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for name in args.dataset:
        if name not in specs:
            raise SystemExit(f"Jeu introuvable: {name}")
        row = train_one(specs[name], save_dir=RESULTS_DIR if args.save else None)
        rows.append(row)
        if args.save:
            (RESULTS_DIR / f"{name}_results.json").write_text(
                json.dumps(row, indent=2, default=str), encoding="utf-8"
            )
        flag = "PASS" if row["acceptable"] else "FAIL"
        print(
            f"\n  [{flag}] {name} auroc={row['test_auroc']:.4f} acc={row['test_accuracy']:.4f} "
            f"cf_valid={row['cf_validity']:.4f}",
            flush=True,
        )

    summary_path = RESULTS_DIR / "summary.json"
    if args.save:
        summary_path.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
