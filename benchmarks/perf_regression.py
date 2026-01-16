"""
Deterministic performance regression checks.

Usage:
  python benchmarks/perf_regression.py --record
  python benchmarks/perf_regression.py --check
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from wordcloud.utils.perf_regression import (
    DEFAULT_CONFIG,
    DEFAULT_RUNS,
    DEFAULT_SEED,
    DEFAULT_STRATEGIES,
    DEFAULT_TOLERANCE,
    compare_to_baseline,
    load_baseline,
    run_benchmark,
    save_baseline,
    serialize_results,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wordcloud performance regression checks")
    parser.add_argument("--record", action="store_true", help="Record baseline timings")
    parser.add_argument("--check", action="store_true", help="Check against baseline timings")
    parser.add_argument("--baseline", type=Path, default=repo_root / "benchmarks" / "perf_baselines.json")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.record and not args.check:
        args.check = True

    results = run_benchmark(DEFAULT_STRATEGIES, seed=args.seed, runs=args.runs, config=DEFAULT_CONFIG)
    payload = {
        "seed": args.seed,
        "runs": args.runs,
        "tolerance": args.tolerance,
        "config": DEFAULT_CONFIG,
        "results": serialize_results(results),
    }

    if args.record:
        save_baseline(args.baseline, payload)
        print(f"Baseline recorded at {args.baseline}")
        return

    baseline = load_baseline(args.baseline)
    tolerance = float(baseline.get("tolerance", args.tolerance))
    failures = compare_to_baseline(baseline, results, tolerance)
    if failures:
        print("Performance regression detected:")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)

    print("Performance check passed")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
