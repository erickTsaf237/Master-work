"""CLI : python -m mega_benchmark"""

from __future__ import annotations

import argparse

from mega_benchmark.config import MegaBenchmarkConfig
from mega_benchmark.runner import run_benchmark


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark multi-datasets RuleConEx vs baselines")
    p.add_argument("--retrain", action="store_true")
    p.add_argument("--skip-amazon", action="store_true")
    p.add_argument("--focus", type=str, default=None)
    p.add_argument("--models", type=str, default=None, help="virgules: ruleconex,mlp,rf,...")
    p.add_argument("--sources", type=str, default=None, help="dlbac,hyconex,hyperlogic,local")
    args = p.parse_args()

    cfg = MegaBenchmarkConfig(
        retrain=args.retrain,
        skip_amazon=args.skip_amazon,
        focus_dataset=args.focus,
    )
    if args.models:
        cfg.models = [m.strip() for m in args.models.split(",") if m.strip()]
    if args.sources:
        cfg.dataset_sources = [s.strip() for s in args.sources.split(",") if s.strip()]

    rows = run_benchmark(cfg)
    ok = sum(1 for r in rows if r.get("status") == "ok")
    print(f"\nTerminé : {ok}/{len(rows)} runs OK")


if __name__ == "__main__":
    main()
