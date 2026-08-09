#!/usr/bin/env python
"""Controlled hybrid weight ablation.

    uv run python scripts/ablate_hybrid.py [--include-drafts]

Deliberately small and pre-declared: a handful of sensible lexical/semantic splits plus
RRF, run against the same corpus, the same gold set and the same index. A large sweep over
a gold set this size would find the weights that fit the noise, and reporting the best of
fifty configurations as "the result" is how a benchmark stops meaning anything.

The production default is NOT changed by this script. If deterministic retrieval wins, it
stays the default — the point is to learn which is better, not to make hybrid win.
"""

from __future__ import annotations

import asyncio
import json
import pathlib
import sys
from datetime import UTC, datetime

from researchagent.container import build_container
from researchagent.evaluation import GoldSet, RetrievalBenchmark
from researchagent.services.retrieval.registry import build_retrieval_arms

GOLD_PATH = pathlib.Path("evaluation/gold/retrieval_v1.json")
RESULTS_DIR = pathlib.Path("evaluation/results")

# (lexical/sparse weight, dense weight). Pre-declared and small; endpoints included so a
# monotone trend is visible if one exists.
WEIGHTS: tuple[tuple[float, float], ...] = (
    (1.0, 0.0),
    (0.7, 0.3),
    (0.5, 0.5),
    (0.3, 0.7),
    (0.0, 1.0),
)


async def main(include_drafts: bool) -> int:
    gold = GoldSet.load(GOLD_PATH)
    container = build_container()
    rows: list[dict[str, object]] = []

    try:
        index = await container.knowledge_indexer.build()
        if not index.succeeded:
            print(f"semantic index unavailable: {index.error}", file=sys.stderr)
            return 1
        print(f"index {index.index_version} | {index.objects_indexed} objects\n")

        base = container.retrieval_config
        combinations = [("weighted", lex, dense) for lex, dense in WEIGHTS]
        combinations.append(("rrf", 1.0, 1.0))
        for strategy, lexical, dense in combinations:
            config = base.model_copy(
                update={
                    "fusion": base.fusion.model_copy(
                        update={
                            "strategy": strategy,
                            "lexical_weight": lexical,
                            "sparse_weight": lexical,
                            "dense_weight": dense,
                        }
                    )
                }
            )
            arms = build_retrieval_arms(
                config,
                container.knowledge_repository,
                container.knowledge_retriever,
                container.embedding_model,
                container.vector_store,
            )
            benchmark = RetrievalBenchmark(
                gold, {"hybrid": arms["hybrid"]}, evidence=container.evidence_repository, limit=10
            )
            report = await benchmark.run(reviewed_only=not include_drafts)
            if not report.arms:
                continue
            metrics = report.arms[0].metrics
            label = f"{strategy}:{lexical:.1f}/{dense:.1f}"
            rows.append(
                {
                    "config": label,
                    "strategy": strategy,
                    "lexical_weight": lexical,
                    "dense_weight": dense,
                    "precision_at_5": metrics.precision_at_k.get(5, 0.0),
                    "recall_at_10": metrics.recall_at_k.get(10, 0.0),
                    "mrr": metrics.mrr,
                    "ndcg_at_5": metrics.ndcg_at_k.get(5, 0.0),
                    "evidence_coverage": metrics.evidence_coverage,
                    "latency_ms": metrics.latency_ms,
                    "degraded_queries": len(report.arms[0].degraded_queries),
                }
            )

        header = f"{'config':<20}{'P@5':>8}{'R@10':>8}{'MRR':>8}{'nDCG@5':>9}{'ms':>9}"
        print(header)
        print("-" * len(header))
        for row in rows:
            print(
                f"{row['config']:<20}{row['precision_at_5']:>8.3f}{row['recall_at_10']:>8.3f}"
                f"{row['mrr']:>8.3f}{row['ndcg_at_5']:>9.3f}{row['latency_ms']:>9.1f}"
            )

        payload = {
            "gold_version": gold.version,
            "reviewed_only": not include_drafts,
            "queries_evaluated": len(gold.reviewed if not include_drafts else gold.queries),
            "index_version": index.index_version,
            "embedding_model": base.embeddings.model,
            "embedding_preprocessing": base.embeddings.preprocessing_version,
            "production_default": base.active_retriever,
            "note": (
                "Ablation only. The production default is not changed by this script; "
                "see config/retrieval.yaml."
            ),
            "configurations": rows,
            "run_at": datetime.now(UTC).isoformat(),
        }
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        destination = RESULTS_DIR / (
            f"ablation_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json"
        )
        destination.write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {destination}")
        return 0
    finally:
        await container.aclose()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main("--include-drafts" in sys.argv)))
