#!/usr/bin/env python
"""Compare retrieval strategies on the gold set.

    uv run python scripts/benchmark_retrieval.py [--include-drafts]

Every arm sees the same queries, the same corpus and the same candidate pool. The only
difference is the retrieval strategy.
"""

from __future__ import annotations

import asyncio
import pathlib
import sys

from researchagent.container import build_container
from researchagent.evaluation import GoldSet, RetrievalBenchmark

GOLD_PATH = pathlib.Path("evaluation/gold/retrieval_v1.json")


async def main(include_drafts: bool) -> int:
    gold = GoldSet.load(GOLD_PATH)
    container = build_container()

    try:
        report = await container.knowledge_indexer.build()
        if not report.succeeded:
            print(f"semantic index unavailable: {report.error}", file=sys.stderr)
            print("semantic and hybrid arms will report as degraded\n", file=sys.stderr)
        else:
            print(
                f"index: {report.objects_indexed} objects, version {report.index_version}, "
                f"embedding {report.embedding_ms:.0f}ms "
                f"({report.embedding_ms_per_object:.1f}ms/object)\n"
            )

        benchmark = RetrievalBenchmark(
            gold,
            container.retrieval_arms,
            evidence=container.evidence_repository,
            limit=10,
        )
        results = await benchmark.run(reviewed_only=not include_drafts)

        status = "REVIEWED" if not include_drafts else "DRAFT (indicative only)"
        print(f"gold {results.gold_version} | {results.queries_evaluated} queries | {status}\n")
        if results.queries_evaluated == 0:
            print("no queries at this review status — nothing to measure")
            return 0

        print(results.table(k=5))
        print()
        for arm in results.arms:
            if arm.degraded_queries:
                print(f"  {arm.arm}: degraded on {len(arm.degraded_queries)} query(ies)")
        print("\nevidence coverage (share of expected evidence reachable):")
        for arm in results.arms:
            print(f"  {arm.arm:<16}{arm.metrics.evidence_coverage:.3f}")
        return 0
    finally:
        await container.aclose()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main("--include-drafts" in sys.argv)))
