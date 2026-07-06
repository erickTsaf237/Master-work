"""

Pipeline DLBAC Amazon pour le nouveau_module.



Defaut (aligne sur baselines sklearn / notebook amazon_classical_baselines) :

  one-hot DLBAC complet (~14k) -> bipolar -> HybridDRNet

  entrainement 2 phases : classification (AUROC) puis contrefactuels legers



Usage :

    python train_nouveau_module_dlbac_amazon_quantile.py --dataset amazon1

    python train_nouveau_module_dlbac_amazon_quantile.py --epochs 50 --single-phase

"""



from __future__ import annotations



import argparse
import importlib
import json

from dataclasses import replace

from pathlib import Path



import numpy as np



from feature_engineering_amazon import build_bean_style_splits

from nouveau_module import HybridDRConfig, HybridDRTrainer

from nouveau_module.binary_metrics import (

    predict_with_grant_threshold,

    summarize_binary_metrics,

    tune_grant_threshold,

)

import nouveau_module.sklearn_baseline as _sklearn_baseline

importlib.reload(_sklearn_baseline)
from nouveau_module.sklearn_baseline import train_histgb_baseline, train_linear_svm_baseline

from prepare_dlbac_datasets import (

    DLBAC_ROOT,

    discover_dlbac_datasets,

    explain_counterfactual_flip,

    format_rule,

)

from train_nouveau_module_dlbac_quantile import build_onehot_splits, build_quantile_splits



ROOT = Path(__file__).resolve().parent

RESULTS_DIR = ROOT / "results" / "nouveau_module_dlbac_amazon"

AMAZON_NAMES = ("amazon1", "amazon2", "amazon3")

DENY_LABEL = 0





def discover_amazon_specs(dlbac_root: Path | None = None):

    specs = discover_dlbac_datasets(dlbac_root)

    return [s for s in specs if s.kind == "real_world" and s.name in AMAZON_NAMES and s.has_train]





def oversample_deny_class(

    x: np.ndarray,

    y: np.ndarray,

    *,

    factor: int,

    seed: int = 42,

) -> tuple[np.ndarray, np.ndarray]:

    if factor <= 1:

        return x, y

    deny_idx = np.where(y == DENY_LABEL)[0]

    if deny_idx.size == 0:

        return x, y

    rng = np.random.default_rng(seed)

    extra_idx = rng.choice(deny_idx, size=deny_idx.size * (factor - 1), replace=True)

    x_out = np.concatenate([x, x[extra_idx]], axis=0)

    y_out = np.concatenate([y, y[extra_idx]], axis=0)

    perm = rng.permutation(len(y_out))

    return x_out[perm], y_out[perm]





def config_for_amazon(

    n_features: int,

    *,

    input_encoding: str,

    phase: int = 1,

    total_epochs: int = 50,

) -> HybridDRConfig:

    """

    Phase 1 : classification seule (cf/flip=0), early stop AUROC.

    Phase 2 : reprise du meilleur checkpoint + pertes CF legeres.

    """

    high_dim = n_features >= 512

    is_onehot = input_encoding == "onehot"



    if is_onehot:

        enc = "bipolar"

        num_rules = 48 if high_dim else 64

        hidden = 96 if high_dim else 128

        n_blocks = 3 if high_dim else 4

    else:

        enc = "auto" if input_encoding in ("bean", "quantile") else input_encoding

        num_rules = 64

        hidden = 128

        n_blocks = 4



    cf_warmup = 0

    cf_l = 0.0

    flip_l = 0.0

    if phase == 2:

        cf_l = 0.06

        flip_l = 0.04



    return HybridDRConfig(

        seed=42,

        epochs=total_epochs,

        batch_size=128,

        lr=1e-3,

        num_rules=num_rules,

        hyper_hidden_dim=hidden,

        cf_hidden_dim=hidden,

        tabresnet_n_blocks=n_blocks,

        tabresnet_dropout=0.1,

        bins_per_feature=4,

        input_encoding=enc,

        use_class_weights=True,

        use_focal_loss=False,

        use_weighted_sampler=False,

        early_stop_metric="auroc",

        cf_warmup_epochs=cf_warmup,

        temperature=0.8,

        cf_lambda=cf_l,

        flip_lambda=flip_l,

        rule_sparsity_lambda=0.002,

    )





def print_class_balance(y_train: np.ndarray, y_test: np.ndarray, class_names: list[str]) -> None:

    for split_name, y in [("train", y_train), ("test", y_test)]:

        counts = np.bincount(y.astype(np.int64), minlength=len(class_names))

        total = counts.sum()

        parts = [f"{class_names[i]}={counts[i]} ({100.0 * counts[i] / total:.1f}%)" for i in range(len(counts))]

        print(f"  {split_name}: {', '.join(parts)}", flush=True)





def evaluate_with_threshold(

    trainer: HybridDRTrainer,

    x: np.ndarray,

    y: np.ndarray,

    *,

    grant_threshold: float | None,

    counterfactuals: bool,

) -> dict:

    metrics = trainer.evaluate(x, y, counterfactuals=counterfactuals, grant_threshold=grant_threshold)

    proba = trainer.predict_proba(x)

    y_pred = predict_with_grant_threshold(proba, grant_threshold) if grant_threshold is not None else (

        np.argmax(proba, axis=1)

    )

    metrics["deny_metrics"] = summarize_binary_metrics(y, y_pred, proba, threshold=grant_threshold)

    return metrics





def load_splits(

    spec,

    *,

    encoding: str,

    max_features: int | None,

    use_cache: bool = True,

):

    if encoding in ("bean", "dry_bean"):

        return build_bean_style_splits(spec, val_size=0.2, random_state=42)

    if encoding == "onehot":

        return build_onehot_splits(

            spec,

            val_size=0.2,

            random_state=42,

            max_features=max_features,

            use_cache=use_cache,

        )

    if encoding == "quantile":

        return build_quantile_splits(spec, val_size=0.2, random_state=42)

    raise ValueError(f"encoding inconnu: {encoding}")





def _fit_hybrid(

    trainer: HybridDRTrainer,

    splits,

    x_train: np.ndarray,

    y_train: np.ndarray,

    *,

    verbose: bool,

    resume: bool = False,

):

    return trainer.fit(

        x_train,

        y_train,

        x_val_cont=splits.x_val,

        y_val=splits.y_val,

        feature_names=splits.feature_names,

        class_names=splits.class_names,

        verbose=verbose,

        resume=resume,

    )





def train_amazon(

    name: str,

    spec,

    *,

    encoding: str = "onehot",

    max_features: int | None = None,

    epochs: int | None = None,

    phase1_epochs: int | None = None,

    phase2_epochs: int | None = None,

    two_phase: bool = True,

    deny_oversample: int = 1,

    save_dir: Path | None = None,

    device: str | None = "auto",

    use_cache: bool = True,

    run_svm_baseline: bool = True,

    verbose: bool = True,

) -> dict:

    splits = load_splits(spec, encoding=encoding, max_features=max_features, use_cache=use_cache)



    x_train = splits.x_train

    y_train = splits.y_train

    if deny_oversample > 1:

        x_train, y_train = oversample_deny_class(x_train, y_train, factor=deny_oversample, seed=42)



    total_epochs = epochs or 50

    p1 = phase1_epochs if phase1_epochs is not None else (int(total_epochs * 0.7) if two_phase else total_epochs)

    p2 = phase2_epochs if phase2_epochs is not None else (total_epochs - p1 if two_phase else 0)



    if verbose:

        print(f"\n--- {name} : pipeline={encoding} ---", flush=True)

        if splits.onehot_dim_full is not None:

            print(

                f"  one-hot plein={splits.onehot_dim_full} -> utilise {x_train.shape[1]} colonnes",

                flush=True,

            )

        elif encoding in ("bean", "dry_bean"):

            print(f"  features style Dry Bean : {x_train.shape[1]} cols", flush=True)

        else:

            print(f"  features quantile : {x_train.shape[1]} cols", flush=True)

        print_class_balance(y_train, splits.y_test, splits.class_names)

        if deny_oversample > 1:

            print(f"  (train apres sur-echantillonnage deny x{deny_oversample})", flush=True)



    baselines: dict = {}

    if run_svm_baseline and encoding == "onehot":

        if verbose:

            print("  [Baseline LinearSVM] entrainement...", flush=True)

        baselines["linear_svm"] = train_linear_svm_baseline(

            x_train, y_train, splits.x_val, splits.y_val, splits.x_test, splits.y_test

        )

        if verbose:

            bl = baselines["linear_svm"]

            print(f"    AUROC={bl['test_auroc']:.4f} deny_f1(tuned)={bl['test_deny_f1_tuned']:.4f}", flush=True)



    if verbose:

        print("  [Baseline HistGB] entrainement...", flush=True)

    baselines["histgb"] = train_histgb_baseline(

        x_train, y_train, splits.x_val, splits.y_val, splits.x_test, splits.y_test

    )

    if verbose:

        bl = baselines["histgb"]

        print(

            f"    AUROC={bl['test_auroc']:.4f} deny_f1(tuned)={bl['test_deny_f1_tuned']:.4f}",

            flush=True,

        )



    if verbose:

        print(f"  PyTorch device     : {device or 'auto'}", flush=True)



    cfg1 = config_for_amazon(

        x_train.shape[1],

        input_encoding=splits.input_encoding,

        phase=1,

        total_epochs=p1,

    )

    trainer = HybridDRTrainer(cfg1, device=device)

    if verbose:

        print(f"  -> utilise          : {trainer.device}", flush=True)

        if two_phase and p2 > 0:

            print(f"  Phase 1/{p1} epochs : classification (AUROC, sans CF)", flush=True)



    result = _fit_hybrid(trainer, splits, x_train, y_train, verbose=verbose)



    if two_phase and p2 > 0:

        if verbose:

            print(f"  Phase 2/{p2} epochs : contrefactuels legers (reprise poids phase 1)", flush=True)

        trainer.config = replace(

            config_for_amazon(

                x_train.shape[1],

                input_encoding=splits.input_encoding,

                phase=2,

                total_epochs=p2,

            )

        )

        result2 = _fit_hybrid(trainer, splits, x_train, y_train, verbose=verbose, resume=True)

        result = TrainingResultMerge(result, result2)



    proba_val = trainer.predict_proba(splits.x_val)

    best_threshold, val_tune = tune_grant_threshold(proba_val, splits.y_val, metric="deny_f1")



    metrics_default = evaluate_with_threshold(

        trainer, splits.x_test, splits.y_test, grant_threshold=None, counterfactuals=True

    )

    metrics_tuned = evaluate_with_threshold(

        trainer, splits.x_test, splits.y_test, grant_threshold=best_threshold, counterfactuals=False

    )



    rules = trainer.export_rules(top_per_rule=4, min_abs_weight=0.05)

    cf_block = metrics_default["counterfactuals"]

    deny_default = metrics_default["deny_metrics"]

    deny_tuned = metrics_tuned["deny_metrics"]



    svm_auroc = baselines.get("linear_svm", {}).get("test_auroc")



    summary = {

        "dataset": name,

        "pipeline_encoding": encoding,

        "onehot_dim_full": splits.onehot_dim_full,

        "model_input_dim": int(x_train.shape[1]),

        "two_phase_training": two_phase and p2 > 0,

        "phase1_epochs": p1,

        "phase2_epochs": p2,

        "deny_oversample_factor": deny_oversample,

        "sklearn_baselines": baselines,

        "binary_dim": int(trainer.model.input_dim_bin) if trainer.model else None,

        "binarizer_mode": trainer.binarizer.mode_,

        "best_val_accuracy": float(result.best_val_accuracy),

        "best_val_auroc": float(result.best_val_auroc),

        "val_threshold_tune": val_tune,

        "test_accuracy": float(metrics_default["accuracy"]),

        "test_auroc": metrics_default.get("auroc_ovr"),

        "test_deny_recall": deny_default["deny_recall"],

        "test_deny_f1": deny_default["deny_f1"],

        "test_balanced_accuracy": deny_default["balanced_accuracy"],

        "test_accuracy_tuned": float(metrics_tuned["accuracy"]),

        "test_deny_recall_tuned": deny_tuned["deny_recall"],

        "test_deny_f1_tuned": deny_tuned["deny_f1"],

        "test_balanced_accuracy_tuned": deny_tuned["balanced_accuracy"],

        "grant_threshold": best_threshold,

        "cf_validity": float(cf_block["validity_cf"]),

        "cf_changed_bits": float(cf_block["changed_bits_mean"]),

        "cf_proximity_l1": float(cf_block["proximity_l1_cont_mean"]),

        "n_rules": len(rules),

        "rules_top10": rules[:10],

        "reference_svm_auroc": svm_auroc,

    }



    idx = 0

    y_true = int(splits.y_test[idx])

    y_target = 1 - y_true

    summary["cf_example"] = explain_counterfactual_flip(

        trainer, splits.x_test, idx, y_target, y_true=y_true

    )



    if verbose:

        print(f"\n=== {name} ({encoding}) ===", flush=True)

        print(f"  Val acc / AUROC    : {summary['best_val_accuracy']:.4f} / {summary['best_val_auroc']:.4f}", flush=True)

        if svm_auroc is not None:

            print(f"  Ref. LinearSVM     : AUROC={svm_auroc:.4f}", flush=True)

        print(f"  Test acc / AUROC   : {summary['test_accuracy']:.4f} / {summary['test_auroc']}", flush=True)

        print(

            f"  Hybride (seuil)    : deny_f1={summary['test_deny_f1_tuned']:.4f} "

            f"bal_acc={summary['test_balanced_accuracy_tuned']:.4f}",

            flush=True,

        )

        print(f"  CF validite        : {summary['cf_validity']:.4f}", flush=True)

        for i, rule in enumerate(rules[:3], start=1):

            print(f"  Regle {i}: {format_rule(rule)}", flush=True)



    if save_dir is not None:

        save_dir.mkdir(parents=True, exist_ok=True)

        out_path = save_dir / f"{name}_results.json"



        def _json_default(obj):

            if isinstance(obj, (np.integer, np.floating)):

                return float(obj) if isinstance(obj, np.floating) else int(obj)

            raise TypeError(type(obj))



        out_path.write_text(json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")

        if verbose:

            print(f"  Sauvegarde : {out_path}", flush=True)



    return summary





class TrainingResultMerge:

    """Combine metriques phase 1 et 2 (garde les meilleures val de phase 2)."""



    def __init__(self, phase1, phase2) -> None:

        self.best_val_accuracy = max(phase1.best_val_accuracy, phase2.best_val_accuracy)

        self.best_val_auroc = max(phase1.best_val_auroc, phase2.best_val_auroc)

        self.history = list(phase1.history) + list(phase2.history)





def parse_args() -> argparse.Namespace:

    p = argparse.ArgumentParser(

        description="Amazon DLBAC -> nouveau_module (one-hot complet, 2 phases, AUROC)"

    )

    p.add_argument("--dataset", nargs="*", choices=AMAZON_NAMES)

    p.add_argument("--all", action="store_true")

    p.add_argument("--dlbac-root", type=Path, default=DLBAC_ROOT)

    p.add_argument("--epochs", type=int, default=50)

    p.add_argument("--phase1-epochs", type=int, default=None)

    p.add_argument("--phase2-epochs", type=int, default=None)

    p.add_argument("--single-phase", action="store_true", help="Desactive la phase 2 (CF)")

    p.add_argument(

        "--encoding",

        choices=("onehot", "bean", "dry_bean", "quantile"),

        default="onehot",

    )

    p.add_argument(

        "--max-features",

        type=int,

        default=0,

        help="0 = one-hot complet (~14k amazon1); N = top variance",

    )

    p.add_argument("--deny-oversample", type=int, default=1)

    p.add_argument("--no-cache", action="store_true")

    p.add_argument("--no-svm-baseline", action="store_true")

    p.add_argument("--no-save", action="store_true")

    p.add_argument("--device", type=str, default="auto")

    return p.parse_args()





def main() -> None:

    args = parse_args()

    specs = discover_amazon_specs(args.dlbac_root)

    by_name = {s.name: s for s in specs}



    if args.all:

        names = list(AMAZON_NAMES)

    elif args.dataset:

        names = list(args.dataset)

    else:

        names = ["amazon1"]



    missing = [n for n in names if n not in by_name]

    if missing:

        raise SystemExit(f"Jeux Amazon introuvables: {missing}")



    max_feat = None if args.max_features <= 0 else args.max_features

    save_dir = None if args.no_save else RESULTS_DIR

    results: list[dict] = []



    for name in names:

        print("\n" + "=" * 72, flush=True)

        print(f"Amazon {name} | encoding={args.encoding}", flush=True)

        print("=" * 72, flush=True)

        row = train_amazon(

            name,

            by_name[name],

            encoding=args.encoding,

            max_features=max_feat,

            epochs=args.epochs,

            phase1_epochs=args.phase1_epochs,

            phase2_epochs=args.phase2_epochs,

            two_phase=not args.single_phase,

            deny_oversample=max(1, args.deny_oversample),

            save_dir=save_dir,

            device=args.device,

            use_cache=not args.no_cache,

            run_svm_baseline=not args.no_svm_baseline,

        )

        skip = ("rules_top10", "cf_example", "val_threshold_tune", "sklearn_baselines")

        results.append({k: v for k, v in row.items() if k not in skip})



    print("\n" + "=" * 72, flush=True)

    print("Recapitulatif", flush=True)

    for r in results:

        ref = r.get("reference_svm_auroc")

        ref_s = f" svm={ref:.4f}" if ref is not None else ""

        print(

            f"  {r['dataset']:10s} hyb_auroc={r['test_auroc']:.4f}{ref_s} "

            f"deny_f1={r['test_deny_f1_tuned']:.4f} acc={r['test_accuracy_tuned']:.4f}",

            flush=True,

        )

    if save_dir:

        print(f"\nJSON : {save_dir}/", flush=True)





if __name__ == "__main__":

    main()


