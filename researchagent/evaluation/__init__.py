"""Retrieval evaluation.

Ground truth is human-reviewed, never model-generated: a benchmark whose labels came from
a language model measures agreement with that model, not retrieval quality.
"""

from researchagent.evaluation.benchmark import (
    ArmResult,
    BenchmarkReport,
    RetrievalBenchmark,
    RunIdentity,
)
from researchagent.evaluation.gold import (
    GoldJudgement,
    GoldQuery,
    GoldSet,
    Relevance,
    ReviewStatus,
)
from researchagent.evaluation.metrics import (
    RetrievalMetrics,
    evaluate,
    mean_metrics,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

__all__ = [
    "ArmResult",
    "BenchmarkReport",
    "GoldJudgement",
    "GoldQuery",
    "GoldSet",
    "Relevance",
    "RetrievalBenchmark",
    "RetrievalMetrics",
    "ReviewStatus",
    "RunIdentity",
    "evaluate",
    "mean_metrics",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
]
