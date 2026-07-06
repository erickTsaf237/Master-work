"""
Entraîne DLBACα ResNet sur tous les jeux DLBAC, sauvegarde les poids et calcule les métriques.

Sorties (par défaut ``results/dlbac_alpha/``) :
  <dataset>/
    model.keras      — modèle Keras complet (architecture + poids)
    config.json      — hyperparamètres et formes d'entrée
    history.json     — loss / accuracy par époque
    metrics.json     — métriques sur le jeu de test
  summary.json       — tableau récapitulatif
  progress.json      — reprise après interruption

Usage (env conda ``hyconex`` avec TensorFlow) :
    python train_dlbac_alpha_all_datasets.py
    python train_dlbac_alpha_all_datasets.py --epochs 30 --verbose
    python train_dlbac_alpha_all_datasets.py --dataset u4k-r4k-auth11k amazon1
    python train_dlbac_alpha_all_datasets.py --retrain
    python train_dlbac_alpha_all_datasets.py --skip-amazon   # synthétiques uniquement (plus rapide)
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_OUT_DIR = ROOT / "results" / "dlbac_alpha"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _load_progress(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    data = json.loads(path.read_text(encoding="utf-8"))
    return set(data.get("completed", []))


def _save_progress(path: Path, completed: set[str]) -> None:
    _atomic_write_json(
        path,
        {
            "completed": sorted(completed),
            "updated_at": _utc_now(),
            "n_completed": len(completed),
        },
    )


def _is_done(dataset_dir: Path, retrain: bool) -> bool:
    if retrain:
        return False
    return (dataset_dir / "metrics.json").is_file() and (dataset_dir / "model.keras").is_file()


def _select_datasets(names: list[str] | None):
    from prepare_dlbac_datasets import discover_dlbac_datasets

    specs = discover_dlbac_datasets()
    if not names:
        return specs
    by_name = {s.name: s for s in specs}
    missing = [n for n in names if n not in by_name]
    if missing:
        raise SystemExit(f"Jeux inconnus : {missing}. Disponibles : {sorted(by_name)}")
    return [by_name[n] for n in names]


def _print_row(row: dict) -> None:
    name = row.get("dataset", row.get("dataset_id", "?"))
    status = row.get("status", "?")
    if status == "ok":
        acc = row.get("accuracy", float("nan"))
        f1 = row.get("f1_macro", float("nan"))
        auc = row.get("auc", float("nan"))
        depth = row.get("resnet_depth", "?")
        elapsed = row.get("elapsed_sec", float("nan"))
        auc_s = f"{auc:.4f}" if isinstance(auc, float) and auc == auc else "n/a"
        print(
            f"  [{status}] {name}  acc={acc:.4f}  f1={f1:.4f}  auc={auc_s}  "
            f"depth={depth}  {elapsed:.0f}s"
        )
    elif status == "skipped":
        print(f"  [skipped] {name} — {row.get('reason', '')}")
    else:
        print(f"  [error] {name} — {row.get('error', 'erreur inconnue')}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DLBACα ResNet sur tous les jeux DLBAC (sauvegarde modèle + métriques)"
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Répertoire de sortie (défaut: {DEFAULT_OUT_DIR})",
    )
    parser.add_argument(
        "--dataset",
        nargs="*",
        default=None,
        help="Sous-ensemble de jeux (ex. u4k-r4k-auth11k amazon1). Défaut : tous.",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Époques (défaut : papier, 30 ou 60)")
    parser.add_argument("--retrain", action="store_true", help="Réentraîner même si déjà terminé")
    parser.add_argument("--verbose", action="store_true", help="Logs Keras détaillés")
    parser.add_argument(
        "--skip-amazon",
        action="store_true",
        help="Exclure amazon1/2/3 (très lents ; protocole papier = synthétiques 4-op)",
    )
    args = parser.parse_args()

    try:
        import tensorflow  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "TensorFlow requis. Activez l'environnement conda hyconex puis relancez.\n"
            f"Détail : {exc}"
        ) from exc

    from dlbac_alpha_baseline.trainer import train_eval_dlbac_alpha

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = out_dir / "progress.json"
    completed = _load_progress(progress_path)

    specs = _select_datasets(args.dataset)
    if args.skip_amazon:
        specs = [s for s in specs if not s.name.startswith("amazon")]

    print(f"DLBACα — {len(specs)} jeu(x) → {out_dir}")
    rows: list[dict] = []

    for spec in specs:
        dataset_dir = out_dir / spec.name
        row_base = {
            "model": "DLBACα-ResNet",
            "dataset": spec.name,
            "dataset_id": f"dlbac/{spec.name}",
            "kind": spec.kind,
            "label_mode": spec.label_mode,
        }

        if spec.train_path is None or not spec.train_path.is_file():
            row = {
                **row_base,
                "status": "skipped",
                "reason": "fichier train .sample manquant",
            }
            rows.append(row)
            _print_row(row)
            continue

        if _is_done(dataset_dir, args.retrain):
            cached = json.loads((dataset_dir / "metrics.json").read_text(encoding="utf-8"))
            row = {**row_base, "status": "ok", "cached": True, **cached}
            rows.append(row)
            completed.add(spec.name)
            print(f"  [cached] {spec.name}  acc={row.get('accuracy', float('nan')):.4f}")
            continue

        print(f"\n--- {spec.name} ({spec.kind}, {spec.label_mode}) ---")
        try:
            row = train_eval_dlbac_alpha(
                spec,
                epochs=args.epochs,
                verbose=args.verbose,
                out_dir=dataset_dir,
            )
            row = {**row_base, **row}
            completed.add(spec.name)
            _save_progress(progress_path, completed)
        except Exception as exc:
            row = {
                **row_base,
                "status": "error",
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            err_path = dataset_dir / "error.json"
            _atomic_write_json(err_path, row)

        rows.append(row)
        _print_row(row)

    summary = {
        "updated_at": _utc_now(),
        "out_dir": str(out_dir),
        "n_total": len(rows),
        "n_ok": sum(1 for r in rows if r.get("status") == "ok"),
        "n_skipped": sum(1 for r in rows if r.get("status") == "skipped"),
        "n_error": sum(1 for r in rows if r.get("status") == "error"),
        "epochs_override": args.epochs,
        "include_amazon": not args.skip_amazon,
        "results": rows,
    }
    _atomic_write_json(out_dir / "summary.json", summary)

    ok = summary["n_ok"]
    err = summary["n_error"]
    skip = summary["n_skipped"]
    print(f"\nTerminé : ok={ok} skipped={skip} error={err} — résumé → {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
