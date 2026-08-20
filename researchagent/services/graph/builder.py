"""Graph construction pipeline.

    KnowledgeRepository -> validated knowledge -> mapping -> validation -> repository

Neo4j is a derived index. The knowledge and evidence repositories remain the source of
truth, and this pipeline can be re-run at any time to rebuild the graph from them.
"""

from __future__ import annotations

import time

from pydantic import BaseModel, Field

from researchagent.core.events import EventBus, EventType, GraphPayload
from researchagent.core.interfaces.graph_repository import GraphRepository, GraphStats
from researchagent.core.interfaces.repositories import KnowledgeRepository, PaperRepository
from researchagent.core.logging import get_logger
from researchagent.models.bundle import Contradiction
from researchagent.models.graph import KnowledgeGraph
from researchagent.models.knowledge import PaperKnowledge
from researchagent.models.paper import Paper
from researchagent.services.evidence.contradictions import ContradictionDetector
from researchagent.services.graph.mapper import GraphMapper
from researchagent.services.graph.validator import GraphValidator

logger = get_logger(__name__)


class GraphBuildReport(BaseModel):
    model_config = {"frozen": True}

    version: str = ""
    papers: int = Field(default=0, ge=0)
    papers_skipped: tuple[str, ...] = ()
    nodes: int = Field(default=0, ge=0)
    edges_proposed: int = Field(default=0, ge=0)
    edges_accepted: int = Field(default=0, ge=0)
    edges_rejected: int = Field(default=0, ge=0)
    rejection_reasons: dict[str, int] = Field(default_factory=dict)
    contradictions: int = Field(default=0, ge=0)
    provenance_coverage: float = 0.0
    stats: GraphStats = Field(default_factory=GraphStats)
    duration_ms: float = 0.0
    succeeded: bool = True
    error: str | None = None


class GraphBuilder:
    """Builds and persists a graph generation from validated knowledge."""

    name = "graph_builder"

    def __init__(
        self,
        knowledge: KnowledgeRepository,
        repository: GraphRepository,
        mapper: GraphMapper | None = None,
        validator: GraphValidator | None = None,
        contradictions: ContradictionDetector | None = None,
        papers: PaperRepository | None = None,
        *,
        event_bus: EventBus | None = None,
    ) -> None:
        self._knowledge = knowledge
        self._repository = repository
        self._mapper = mapper or GraphMapper()
        self._validator = validator or GraphValidator()
        self._contradictions = contradictions or ContradictionDetector()
        self._papers = papers
        self._event_bus = event_bus

    async def build(
        self, paper_ids: tuple[str, ...] = (), *, run_id: str | None = None
    ) -> GraphBuildReport:
        started = time.perf_counter()

        knowledge, skipped = await self._collect(paper_ids)
        if not knowledge:
            return GraphBuildReport(
                papers_skipped=skipped,
                succeeded=False,
                error="no validated knowledge available",
                duration_ms=_ms(started),
            )

        contradictions = self._detect_contradictions(knowledge)
        papers = await self._papers_for(knowledge)
        graph = self._mapper.build(knowledge, papers=papers, contradictions=contradictions)
        report = self._validator.validate(graph)

        # Only validated edges are persisted; rejections are reported, never hidden.
        accepted = KnowledgeGraph(
            version=graph.version, nodes=graph.nodes, edges=report.accepted_edges
        )
        stats = await self._repository.write_graph(accepted)

        build_report = GraphBuildReport(
            version=graph.version.identifier,
            papers=len(knowledge),
            papers_skipped=skipped,
            nodes=len(accepted.nodes),
            edges_proposed=len(graph.edges),
            edges_accepted=len(accepted.edges),
            edges_rejected=len(report.rejected_edges),
            rejection_reasons=report.rejection_reasons,
            contradictions=len(contradictions),
            provenance_coverage=round(accepted.provenance_coverage, 4),
            stats=stats,
            duration_ms=_ms(started),
            succeeded=report.result.success,
            error=None if report.result.success else "; ".join(report.result.issue_codes()),
        )
        logger.info(
            "graph_built",
            version=build_report.version,
            nodes=build_report.nodes,
            edges=build_report.edges_accepted,
            rejected=build_report.edges_rejected,
            contradictions=build_report.contradictions,
        )
        await self._emit(build_report, run_id)
        return build_report

    def _detect_contradictions(self, knowledge: list[PaperKnowledge]) -> tuple[Contradiction, ...]:
        """Detected across the whole corpus: cross-paper disagreement is the interesting kind."""
        every_object = tuple(obj for item in knowledge for obj in item.objects)
        return self._contradictions.detect(every_object)

    async def _collect(
        self, paper_ids: tuple[str, ...]
    ) -> tuple[list[PaperKnowledge], tuple[str, ...]]:
        wanted = paper_ids or tuple(
            key.replace("-", ":", 1) for key in await self._knowledge.list_ids()
        )
        collected: list[PaperKnowledge] = []
        skipped: list[str] = []

        for paper_id in wanted:
            stored = await self._knowledge.get(paper_id)
            if stored is None or not stored.is_trusted:
                # Zero trust: knowledge the previous stage rejected never becomes a node.
                skipped.append(paper_id)
                continue
            collected.append(stored.value)
        return collected, tuple(skipped)

    async def _papers_for(self, knowledge: list[PaperKnowledge]) -> dict[str, Paper]:
        if self._papers is None:
            return {}
        found: dict[str, Paper] = {}
        for item in knowledge:
            record = await self._papers.get(item.paper_id)
            if record is not None:
                found[item.paper_id] = record.paper
        return found

    async def _emit(self, report: GraphBuildReport, run_id: str | None) -> None:
        if self._event_bus is None:
            return
        await self._event_bus.emit(
            EventType.GRAPH_BUILT,
            GraphPayload(
                graph_version=report.version,
                nodes=report.nodes,
                edges=report.edges_accepted,
                rejected_edges=report.edges_rejected,
                contradictions=report.contradictions,
                provenance_coverage=report.provenance_coverage,
            ),
            run_id=run_id,
            source=self.name,
        )


def _ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 3)
