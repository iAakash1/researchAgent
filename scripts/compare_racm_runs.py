#!/usr/bin/env python
"""Compare two retraction-benchmark reports.

    uv run python scripts/compare_racm_runs.py smoke smoke_rerun

Refuses when the reports disagree on topic, corpus fingerprint, configuration digest,
seed, K, flip statuses or grounding floor: numbers from different setups side by side
would imply a comparison that was never made.
"""

from __future__ import annotations

import pathlib
import sys

from researchagent.evaluation.retraction import BenchmarkReport, comparable

RESULTS_DIR = pathlib.Path("evaluation/results")
FIELDS = (
    ("invalidation_precision", "invalidation precision"),
    ("invalidation_recall", "invalidation recall"),
    ("over_invalidation_rate", "over-invalidation rate"),
    ("wrongful_invalidations", "wrongful invalidations"),
    ("needs_review_correctness", "needs-review correctness"),
    ("leakage_rate", "retracted-source leakage"),
)


def main(labels: list[str]) -> int:
    if len(labels) != 2:
        print("usage: compare_racm_runs.py <label_a> <label_b>", file=sys.stderr)
        return 2
    left, right = (
        BenchmarkReport.model_validate_json((RESULTS_DIR / f"racm_{label}.json").read_text())
        for label in labels
    )
    mismatched = comparable(left, right)
    if mismatched:
        print(f"NOT COMPARABLE — identities differ on: {list(mismatched)}", file=sys.stderr)
        return 1

    strategies = [result.strategy for result in left.results]
    for strategy in strategies:
        a, b = left.result(strategy), right.result(strategy)
        if a is None or b is None:
            continue
        print(f"\n{strategy.value}")
        for key, name in FIELDS:
            print(f"  {name:<28}{getattr(a.metrics, key)!s:>12}{getattr(b.metrics, key)!s:>12}")
        print(f"  {'total tokens':<28}{a.usage.total_tokens:>12}{b.usage.total_tokens:>12}")
        cost_a, cost_b = a.usage.estimated_cost_usd, b.usage.estimated_cost_usd
        print(f"  {'cost (USD)':<28}{cost_a!s:>12}{cost_b!s:>12}")
        print(f"  {'latency (ms)':<28}{a.usage.latency_ms:>12.0f}{b.usage.latency_ms:>12.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
