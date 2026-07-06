"""Boucle benchmark avec reprise (checkpoint JSON par dataset × modèle)."""

from __future__ import annotations

import json
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mega_benchmark.config import MegaBenchmarkConfig
from mega_benchmark.datasets import discover_all_dataset_ids, load_splits


def _model_registry(*, reload: bool = False) -> tuple[dict, dict]:
    """Charge MODEL_RUNNERS (reload=True après mise à jour du code sans redémarrer le kernel)."""
    import importlib

    import mega_benchmark.models as models_mod

    if reload:
        importlib.reload(models_mod)
    return models_mod.MODEL_RUNNERS, models_mod.MODEL_LABELS


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _result_path(results_dir: Path, model_key: str, dataset_id: str) -> Path:
    safe = dataset_id.replace("/", "__").replace(" ", "_")
    return results_dir / model_key / f"{safe}_results.json"


def _load_progress(path: Path) -> set[tuple[str, str]]:
    if not path.is_file():
        return set()
    data = json.loads(path.read_text(encoding="utf-8"))
    return {tuple(x) for x in data.get("completed", [])}


def _save_progress(path: Path, completed: set[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "completed": sorted([list(x) for x in completed]),
                "updated_at": _utc_now(),
                "n_completed": len(completed),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def run_one(
    dataset_id: str,
    model_key: str,
    cfg: MegaBenchmarkConfig,
    *,
    results_dir: Path,
    model_runners: dict | None = None,
    model_labels: dict | None = None,
) -> dict[str, Any]:
    runners, labels = model_runners, model_labels
    if runners is None or labels is None:
        runners, labels = _model_registry()
    if model_key not in runners:
        raise KeyError(
            f"Modèle inconnu: {model_key}. Disponibles: {sorted(runners)}"
        )

    splits = load_splits(dataset_id, seed=cfg.seed)
    out_path = _result_path(results_dir, model_key, dataset_id)

    if not cfg.retrain and out_path.is_file():
        row = json.loads(out_path.read_text(encoding="utf-8"))
        row["_from_cache"] = True
        return row

    try:
        row = runners[model_key](splits, cfg)
        row["model_key"] = model_key
        row["model_label"] = labels.get(model_key, model_key)
        row["dataset_id"] = dataset_id
        row["finished_at"] = _utc_now()
        row["_from_cache"] = False
    except Exception as exc:
        row = {
            "model_key": model_key,
            "model_label": labels.get(model_key, model_key),
            "dataset_id": dataset_id,
            "status": "error",
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "finished_at": _utc_now(),
            "_from_cache": False,
        }

    _atomic_write_json(out_path, row)
    return row


def run_benchmark(cfg: MegaBenchmarkConfig | None = None) -> list[dict[str, Any]]:
    cfg = cfg or MegaBenchmarkConfig()
    model_runners, model_labels = _model_registry(reload=True)

    unknown = [m for m in cfg.models if m not in model_runners]
    if unknown:
        raise ValueError(
            f"Modèles inconnus: {unknown}. Disponibles: {sorted(model_runners)}"
        )

    results_dir = Path(cfg.results_dir)
    if not results_dir.is_absolute():
        results_dir = Path(__file__).resolve().parent.parent / results_dir

    progress_path = results_dir / "progress.json"
    completed = _load_progress(progress_path)

    dataset_ids = discover_all_dataset_ids(
        sources=cfg.dataset_sources,
        skip_amazon=cfg.skip_amazon,
    )
    if cfg.focus_dataset:
        dataset_ids = [d for d in dataset_ids if d == cfg.focus_dataset or d.endswith(cfg.focus_dataset)]

    rows: list[dict[str, Any]] = []
    total = len(dataset_ids) * len(cfg.models)
    done = 0

    for dataset_id in dataset_ids:
        for model_key in cfg.models:
            pair = (dataset_id, model_key)
            out_path = _result_path(results_dir, model_key, dataset_id)

            if not cfg.retrain and pair in completed and out_path.is_file():
                row = json.loads(out_path.read_text(encoding="utf-8"))
                row["_from_cache"] = True
                rows.append(row)
                done += 1
                if cfg.verbose:
                    print(f"[cache {done}/{total}] {dataset_id} | {model_key}", flush=True)
                continue

            if cfg.verbose:
                print(f"[run {done + 1}/{total}] {dataset_id} | {model_key}", flush=True)

            row = run_one(
                dataset_id,
                model_key,
                cfg,
                results_dir=results_dir,
                model_runners=model_runners,
                model_labels=model_labels,
            )
            rows.append(row)

            if row.get("status") in ("ok", "skipped"):
                completed.add(pair)
                _save_progress(progress_path, completed)

            done += 1

    summary_path = results_dir / "summary.json"
    _atomic_write_json(
        summary_path,
        {
            "updated_at": _utc_now(),
            "config": cfg.__dict__,
            "n_rows": len(rows),
            "rows": [{k: v for k, v in r.items() if k != "traceback"} for r in rows],
        },
    )
    build_pivot_csv(results_dir)
    return rows


def build_pivot_csv(results_dir: Path) -> Path:
    import csv

    rows: list[dict[str, Any]] = []
    for model_dir in sorted(results_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        for jf in sorted(model_dir.glob("*_results.json")):
            try:
                rows.append(json.loads(jf.read_text(encoding="utf-8")))
            except Exception:
                pass

    csv_path = results_dir / "metrics_pivot.csv"
    if not rows:
        return csv_path

    fieldnames = [
        "dataset_id",
        "model_key",
        "model_label",
        "status",
        "accuracy",
        "f1_macro",
        "auc",
        "cf_validity",
        "binary_accuracy",
        "elapsed_sec",
        "device",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return csv_path


def load_summary(results_dir: Path | str) -> list[dict[str, Any]]:
    p = Path(results_dir)
    summary = p / "summary.json"
    if summary.is_file():
        return json.loads(summary.read_text(encoding="utf-8")).get("rows", [])
    rows: list[dict[str, Any]] = []
    for model_dir in sorted(p.iterdir()):
        if not model_dir.is_dir():
            continue
        for jf in sorted(model_dir.glob("*_results.json")):
            rows.append(json.loads(jf.read_text(encoding="utf-8")))
    return rows
