"""
Lance dlbac_alpha_resnet.py (script officiel du ReadMe) sur tous les jeux synthétiques.

Reproduit la commande documentée :
    python3 dlbac_alpha_resnet.py <train.sample> <test.sample>

Pour chaque jeu, les sorties officielles (results/dlbac_alpha.hdf5, history_*, result.txt)
sont copiées dans results/<nom_du_jeu>/ pour éviter l'écrasement.

Usage (depuis ce dossier, env hyconex) :
    python run_all_synthetic_datasets.py
    python run_all_synthetic_datasets.py --dataset u4k-r4k-auth11k
    python run_all_synthetic_datasets.py --retrain
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
DLBAC_ROOT = HERE.parent / "dataset"
OFFICIAL_SCRIPT = HERE / "dlbac_alpha_resnet.py"
RESULTS_ROOT = HERE / "results"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _discover_synthetic() -> list[tuple[str, Path, Path]]:
    syn_root = DLBAC_ROOT / "synthetic"
    if not syn_root.is_dir():
        raise SystemExit(f"Dossier introuvable : {syn_root}")

    out: list[tuple[str, Path, Path]] = []
    for folder in sorted(syn_root.iterdir()):
        if not folder.is_dir():
            continue
        name = folder.name
        train = folder / f"train_{name}.sample"
        test = folder / f"test_{name}.sample"
        if not train.is_file() or not test.is_file():
            continue
        out.append((name, train, test))
    return out


def _official_outputs_exist(dataset_dir: Path) -> bool:
    return (
        (dataset_dir / "dlbac_alpha.hdf5").is_file()
        and (dataset_dir / "result.txt").is_file()
    )


def _archive_run(dataset_name: str) -> Path:
    """Copie results/ officiel vers results/<dataset_name>/."""
    src = HERE / "results"
    dst = RESULTS_ROOT / dataset_name
    dst.mkdir(parents=True, exist_ok=True)

    for fname in ("dlbac_alpha.hdf5", "result.txt"):
        f = src / fname
        if f.is_file():
            shutil.copy2(f, dst / fname)

    history_src = src / "history_dlbac_alpha"
    if history_src.is_file():
        shutil.copy2(history_src, dst / "history_dlbac_alpha")

    meta = {
        "dataset": dataset_name,
        "archived_at": _utc_now(),
        "files": sorted(p.name for p in dst.iterdir() if p.is_file()),
    }
    (dst / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return dst


def _run_one(train: Path, test: Path, *, verbose: bool) -> int:
    cmd = [sys.executable, str(OFFICIAL_SCRIPT), str(train), str(test)]
    print(f"  $ {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=HERE, check=False)
    return proc.returncode


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DLBACα officiel (dlbac_alpha_resnet.py) sur tous les jeux synthétiques"
    )
    parser.add_argument(
        "--dataset",
        nargs="*",
        default=None,
        help="Sous-ensemble (ex. u4k-r4k-auth11k u4k-r4k-auth21k)",
    )
    parser.add_argument("--retrain", action="store_true", help="Réentraîner même si déjà archivé")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if not OFFICIAL_SCRIPT.is_file():
        raise SystemExit(f"Script officiel introuvable : {OFFICIAL_SCRIPT}")

    datasets = _discover_synthetic()
    if args.dataset:
        wanted = set(args.dataset)
        datasets = [d for d in datasets if d[0] in wanted]
        missing = wanted - {d[0] for d in datasets}
        if missing:
            raise SystemExit(f"Jeux sans train+test : {sorted(missing)}")

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    summary: list[dict] = []

    print(f"DLBACα officiel — {len(datasets)} jeu(x) synthétique(s)")
    for name, train, test in datasets:
        dataset_dir = RESULTS_ROOT / name
        row = {"dataset": name, "train": str(train), "test": str(test)}

        if _official_outputs_exist(dataset_dir) and not args.retrain:
            row["status"] = "cached"
            summary.append(row)
            print(f"\n[cached] {name}")
            continue

        print(f"\n=== {name} ===")
        code = _run_one(train, test, verbose=args.verbose)
        if code != 0:
            row["status"] = "error"
            row["exit_code"] = code
            summary.append(row)
            print(f"  ERREUR (code {code})")
            continue

        archived = _archive_run(name)
        row["status"] = "ok"
        row["output_dir"] = str(archived)
        summary.append(row)
        print(f"  OK → {archived}")

    summary_path = RESULTS_ROOT / "summary.json"
    payload = {
        "updated_at": _utc_now(),
        "n_ok": sum(1 for r in summary if r["status"] == "ok"),
        "n_cached": sum(1 for r in summary if r["status"] == "cached"),
        "n_error": sum(1 for r in summary if r["status"] == "error"),
        "runs": summary,
    }
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"\nTerminé : ok={payload['n_ok']} cached={payload['n_cached']} "
        f"error={payload['n_error']} — {summary_path}"
    )


if __name__ == "__main__":
    main()
