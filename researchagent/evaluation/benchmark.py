"""Retrieval benchmark.

Runs several retrievers over the same gold set, the same corpus and the same candidate
pool, so the only thing that differs is the retrieval strategy.

Reports what it measured, including when the answer is "the baseline won". The purpose is
to find out whether semantic retrieval helps on this corpus, not to demonstrate that it
does.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime

from pydantic import BaseModel, Field

from researchagent.core.interfaces.evidence_repository import EvidenceRepository
from researchagent.core.interfaces.retrieval import KnowledgeRetriever
from researchagent.core.logging import get_logger
from researchagent.evaluation.gold import GoldQuery, GoldSet
from researchagent.evaluation.metrics import RetrievalMetrics, evaluate, mean_metrics
from researchagent.models.query import ResearchQuery

logger = get_logger(__name__)


class ArmResult(BaseModel):
    """One retriever's results across the whole gold set."""

    model_config = {"frozen": True}

    arm: str
    metrics: RetrievalMetrics
    per_query: dict[str, RetrievalMetrics] = Field(default_factory=dict)
    degraded_queries: tuple[str, ...] = ()
    total_ms: float = 0.0

    @property
    def is_complete(self) -> bool:
        return not self.degraded_queries


class RunIdentity(BaseModel):
    """Everything needed to reproduce a benchmark number, or to refuse to compare two.

    A metric without this is uninterpretable: "hybrid scored 0.41" says nothing unless the
    corpus, the judgements, the embedding model and the index it ran against are all
    pinned. Two reports whose identities differ are not comparable, and recording the
    identity is what makes that checkable rather than assumed.
    """

    model_config = {"frozen": True}

    corpus_version: str = Field(description="Fingerprint of the papers and objects indexed")
    papers: int = 0
    knowledge_objects: int = 0
    evidence_objects: int = 0
    gold_version: str = ""
    gold_reviewed: int = 0
    gold_drafts: int = 0
    embedding_model: str = ""
    embedding_preprocessing: str = ""
    index_version: str = ""
    retrieval_config: dict[str, object] = Field(default_factory=dict)
    ks: tuple[int, ...] = ()
    limit: int = 0


class BenchmarkReport(BaseModel):
    """A reproducible comparison. Cite the gold version alongside any number from it."""

    model_config = {"frozen": True}

    gold_version: str
    queries_evaluated: int
    reviewed_only: bool
    arms: tuple[ArmResult, ...] = ()
    identity: RunIdentity | None = None
    run_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def is_citable(self) -> bool:
        """Whether a claim may be made from these numbers.

        Draft judgements produce indicative numbers only: they were proposed from the
        corpus, not confirmed by a person, so a claim resting on them would be the system
        grading its own homework.
        """
        return self.reviewed_only and self.queries_evaluated > 0

    def table(self, k: int = 5) -> str:
        header = (
            f"{'arm':<16}{'P@' + str(k):>8}{'R@10':>8}{'MRR':>8}{'nDCG@' + str(k):>10}{'ms':>9}"
        )
        rows = [header, "-" * len(header)]
        for arm in self.arms:
            metrics = arm.metrics
            rows.append(
                f"{arm.arm:<16}"
                f"{metrics.precision_at_k.get(k, 0.0):>8.3f}"
                f"{metrics.recall_at_k.get(10, 0.0):>8.3f}"
                f"{metrics.mrr:>8.3f}"
                f"{metrics.ndcg_at_k.get(k, 0.0):>10.3f}"
                f"{metrics.latency_ms:>9.1f}"
            )
        return "\n".join(rows)


class RetrievalBenchmark:
    """Runs the same queries through several retrievers."""

    name = "retrieval_benchmark"

    def __init__(
        self,
        gold: GoldSet,
        arms: dict[str, KnowledgeRetriever],
        *,
        evidence: EvidenceRepository | None = None,
        limit: int = 10,
        ks: tuple[int, ...] = (1, 3, 5, 10),
        identity: RunIdentity | None = None,
    ) -> None:
        self._gold = gold
        self._arms = arms
        self._evidence = evidence
        self._limit = limit
        self._ks = ks
        self._identity = identity

    async def run(self, *, reviewed_only: bool = True) -> BenchmarkReport:
        queries = self._gold.reviewed if reviewed_only else self._gold.queries
        if not queries:
            logger.warning(
                "benchmark_no_queries",
                reviewed_only=reviewed_only,
                drafts=len(self._gold.drafts),
            )

        arms = [
            await self._run_arm(name, retriever, queries) for name, retriever in self._arms.items()
        ]

        identity = self._identity
        if identity is not None:
            identity = identity.model_copy(
                update={
                    "gold_version": self._gold.version,
                    "gold_reviewed": len(self._gold.reviewed),
                    "gold_drafts": len(self._gold.drafts),
                    "ks": self._ks,
                    "limit": self._limit,
                }
            )

        return BenchmarkReport(
            gold_version=self._gold.version,
            queries_evaluated=len(queries),
            reviewed_only=reviewed_only,
            arms=tuple(arms),
            identity=identity,
        )

    async def _run_arm(
        self, name: str, retriever: KnowledgeRetriever, queries: tuple[GoldQuery, ...]
    ) -> ArmResult:
        started = time.perf_counter()
        per_query: dict[str, RetrievalMetrics] = {}
        degraded: list[str] = []

        for gold_query in queries:
            query = ResearchQuery(
                text=gold_query.text,
                kinds=gold_query.kinds,
                paper_ids=gold_query.paper_ids,
                limit=self._limit,
            )
            result = await retriever.retrieve(query)
            if result.degraded:
                degraded.append(gold_query.id)

            retrieved = [hit.item.id for hit in result.hits]
            per_query[gold_query.id] = evaluate(
                retrieved,
                gold_query.relevant_ids,
                gold_query.grades,
                ks=self._ks,
                evidence_coverage=await self._evidence_coverage(gold_query, retrieved),
                latency_ms=result.latency_ms,
            )

        return ArmResult(
            arm=name,
            metrics=mean_metrics(list(per_query.values())),
            per_query=per_query,
            degraded_queries=tuple(degraded),
            total_ms=round((time.perf_counter() - started) * 1000, 3),
        )

    async def _evidence_coverage(self, gold_query: GoldQuery, retrieved: list[str]) -> float:
        """Share of the expected evidence reachable from what was retrieved.

        The metric closest to what the system is for: ranking ids well is only useful if
        the evidence behind them can be assembled into a bundle.
        """
        if not gold_query.relevant_evidence_ids or self._evidence is None:
            return 0.0
        records = await self._evidence.for_objects(tuple(retrieved))
        found = {record.id for record in records}
        expected = set(gold_query.relevant_evidence_ids)
        return len(found & expected) / len(expected)
