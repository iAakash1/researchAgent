#!/usr/bin/env python
"""Compare retrieval strategies on the gold set.

    uv run python scripts/benchmark_retrieval.py [--include-drafts] [--out PATH]

Every arm sees the same queries, the same corpus and the same candidate pool. The only
difference is the retrieval strategy.

Results are written as JSON carrying the full run identity — corpus fingerprint, gold
version, embedding model, index version, retrieval config, timestamp — because a metric
without those is not reproducible and two reports with different identities are not
comparable.
"""

from __future__ import annotations

import asyncio
import hashlib
import pathlib
import sys
from datetime import UTC, datetime

from researchagent.container import Container, build_container
from researchagent.evaluation import GoldSet, RetrievalBenchmark, RunIdentity

GOLD_PATH = pathlib.Path("evaluation/gold/retrieval_v1.json")
RESULTS_DIR = pathlib.Path("evaluation/results")


async def corpus_identity(container: Container, index_version: str) -> RunIdentity:
    """Fingerprint what the benchmark actually ran against."""
    paper_ids = sorted(await container.knowledge_repository.list_ids())
    objects = 0
    evidence = 0
    for paper_id in paper_ids:
        knowledge = await container.knowledge_repository.get(paper_id)
        objects += len(knowledge.value.objects) if knowledge else 0
        stored = await container.evidence_repository.get_paper(paper_id)
        evidence += len(stored.records) if stored else 0

    payload = "|".join(paper_ids) + f"#{objects}"
    retrieval = container.retrieval_config
    return RunIdentity(
        corpus_version=hashlib.sha256(payload.encode()).hexdigest()[:16],
        papers=len(paper_ids),
        knowledge_objects=objects,
        evidence_objects=evidence,
        embedding_model=retrieval.embeddings.model,
        embedding_preprocessing=retrieval.embeddings.preprocessing_version,
        index_version=index_version,
        retrieval_config={
            "active_retriever": retrieval.active_retriever,
            "fusion_strategy": retrieval.fusion.strategy,
            "lexical_weight": retrieval.fusion.lexical_weight,
            "sparse_weight": retrieval.fusion.sparse_weight,
            "dense_weight": retrieval.fusion.dense_weight,
            "candidate_multiplier": retrieval.fusion.candidate_multiplier,
            "rrf_k": retrieval.fusion.rrf_k,
            "bm25_k1": retrieval.bm25.k1,
            "bm25_b": retrieval.bm25.b,
        },
    )


async def main(include_drafts: bool, out: pathlib.Path | None) -> int:
    gold = GoldSet.load(GOLD_PATH)
    container = build_container()

    try:
        index = await container.knowledge_indexer.build()
        if not index.succeeded:
            print(f"semantic index unavailable: {index.error}", file=sys.stderr)
            print("semantic and hybrid arms will report as degraded\n", file=sys.stderr)
        else:
            print(
                f"index: {index.objects_indexed} objects, version {index.index_version}, "
                f"embedding {index.embedding_ms:.0f}ms "
                f"({index.embedding_ms_per_object:.1f}ms/object)\n"
            )

        identity = await corpus_identity(container, index.index_version)
        benchmark = RetrievalBenchmark(
            gold,
            container.retrieval_arms,
            evidence=container.evidence_repository,
            limit=10,
            identity=identity,
        )
        results = await benchmark.run(reviewed_only=not include_drafts)

        status = "REVIEWED" if not include_drafts else "DRAFT (indicative only)"
        print(
            f"corpus {identity.corpus_version} | {identity.papers} papers | "
            f"{identity.knowledge_objects} objects | {identity.evidence_objects} evidence"
        )
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

        if not results.is_citable:
            print("\nNOTE: draft judgements. Indicative only — not a claim about retrieval.")

        destination = out or RESULTS_DIR / (
            f"benchmark_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(results.model_dump_json(indent=2))
        print(f"\nwrote {destination}")
        return 0
    finally:
        await container.aclose()


if __name__ == "__main__":
    argv = sys.argv[1:]
    out_path = None
    if "--out" in argv:
        out_path = pathlib.Path(argv[argv.index("--out") + 1])
    raise SystemExit(asyncio.run(main("--include-drafts" in argv, out_path)))
