"""Retrieval metrics.

Standard IR measures, implemented directly so the benchmark has no hidden behaviour and
the numbers can be recomputed by hand from a result list.

Graded relevance where the gold set provides it: an object that fully answers a question
and one that is merely related should not contribute equally, and nDCG is the measure
that respects that.
"""

from __future__ import annotations

import math

from pydantic import BaseModel, Field


class RetrievalMetrics(BaseModel):
    """One retriever's performance on one query."""

    model_config = {"frozen": True}

    precision_at_k: dict[int, float] = Field(default_factory=dict)
    recall_at_k: dict[int, float] = Field(default_factory=dict)
    ndcg_at_k: dict[int, float] = Field(default_factory=dict)
    mrr: float = 0.0
    # Share of the gold evidence reachable from the retrieved objects. The measure that
    # matters most here: retrieval exists to assemble evidence, not to rank ids.
    evidence_coverage: float = 0.0
    retrieved: int = 0
    relevant_available: int = 0
    latency_ms: float = 0.0


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if k <= 0 or not retrieved:
        return 0.0
    window = retrieved[:k]
    return sum(1 for item in window if item in relevant) / len(window)


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return sum(1 for item in retrieved[:k] if item in relevant) / len(relevant)


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    """1/rank of the first relevant hit. Rewards putting something right at the top."""
    for rank, item in enumerate(retrieved, start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[str], grades: dict[str, float], k: int) -> float:
    """Normalised discounted cumulative gain with graded relevance."""
    if not grades or k <= 0:
        return 0.0

    gain = sum(
        grades.get(item, 0.0) / math.log2(rank + 1)
        for rank, item in enumerate(retrieved[:k], start=1)
    )
    ideal_grades = sorted(grades.values(), reverse=True)[:k]
    ideal = sum(grade / math.log2(rank + 1) for rank, grade in enumerate(ideal_grades, start=1))
    return gain / ideal if ideal > 0 else 0.0


def evaluate(
    retrieved: list[str],
    relevant: set[str],
    grades: dict[str, float],
    *,
    ks: tuple[int, ...] = (1, 3, 5, 10),
    evidence_coverage: float = 0.0,
    latency_ms: float = 0.0,
) -> RetrievalMetrics:
    return RetrievalMetrics(
        precision_at_k={k: round(precision_at_k(retrieved, relevant, k), 4) for k in ks},
        recall_at_k={k: round(recall_at_k(retrieved, relevant, k), 4) for k in ks},
        ndcg_at_k={k: round(ndcg_at_k(retrieved, grades, k), 4) for k in ks},
        mrr=round(reciprocal_rank(retrieved, relevant), 4),
        evidence_coverage=round(evidence_coverage, 4),
        retrieved=len(retrieved),
        relevant_available=len(relevant),
        latency_ms=round(latency_ms, 3),
    )


def mean_metrics(results: list[RetrievalMetrics]) -> RetrievalMetrics:
    """Macro-average across queries: every question counts equally regardless of size."""
    if not results:
        return RetrievalMetrics()

    ks = sorted({k for result in results for k in result.precision_at_k})
    count = len(results)

    def mean(values: list[float]) -> float:
        return round(sum(values) / count, 4)

    return RetrievalMetrics(
        precision_at_k={k: mean([r.precision_at_k.get(k, 0.0) for r in results]) for k in ks},
        recall_at_k={k: mean([r.recall_at_k.get(k, 0.0) for r in results]) for k in ks},
        ndcg_at_k={k: mean([r.ndcg_at_k.get(k, 0.0) for r in results]) for k in ks},
        mrr=mean([r.mrr for r in results]),
        evidence_coverage=mean([r.evidence_coverage for r in results]),
        retrieved=sum(r.retrieved for r in results),
        relevant_available=sum(r.relevant_available for r in results),
        latency_ms=mean([r.latency_ms for r in results]),
    )
