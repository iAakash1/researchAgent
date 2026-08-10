"""The concrete research toolbox.

Composes the retrieval, evidence and graph services an agent may reach, and nothing else.
There is no filesystem access, no shell, no database handle and no query language here —
an agent chooses among seven domain verbs, each of which validates its own arguments and
caps its own results.

Every call is timed and recorded, successes and failures alike, because "the graph was
unavailable when this conclusion was drawn" is part of how the conclusion was reached.
"""

from __future__ import annotations

import time
from typing import ClassVar

from researchagent.core.constants import SECONDS_PER_MILLISECOND
from researchagent.core.exceptions import ResearchAgentError
from researchagent.core.interfaces.graph_repository import GraphRepository
from researchagent.core.interfaces.retrieval import KnowledgeRetriever
from researchagent.core.interfaces.tools import (
    EvidenceSearchResult,
    GraphSearchResult,
    KnowledgeSearchResult,
    PaperContext,
    ResearchToolbox,
    ToolCall,
    ToolName,
)
from researchagent.core.logging import get_logger
from researchagent.models.bundle import Contradiction, EvidenceBundle
from researchagent.models.knowledge import KnowledgeKind, KnowledgeObject
from researchagent.models.query import QueryIntent, ResearchQuery
from researchagent.repositories.bundle_repository import JsonBundleRepository
from researchagent.repositories.evidence_repository import JsonEvidenceRepository
from researchagent.repositories.knowledge_repository import JsonKnowledgeRepository
from researchagent.repositories.paper_repository import JsonPaperRepository
from researchagent.services.evidence.pipeline import EvidenceIntelligenceService
from researchagent.services.graph.queries import GraphQueries

logger = get_logger(__name__)

# Hard ceilings. An agent may ask for less; it cannot ask for more, whatever it emits.
MAX_LIMIT = 50
MAX_DEPTH = 3
MAX_IDS = 50
MAX_QUERY_CHARS = 500


class ServiceToolbox(ResearchToolbox):
    """Domain verbs over the existing services."""

    name: ClassVar[str] = "service_toolbox"

    def __init__(
        self,
        retriever: KnowledgeRetriever,
        evidence_service: EvidenceIntelligenceService,
        knowledge_repository: JsonKnowledgeRepository,
        evidence_repository: JsonEvidenceRepository,
        paper_repository: JsonPaperRepository,
        bundle_repository: JsonBundleRepository | None = None,
        graph_repository: GraphRepository | None = None,
        graph_queries: GraphQueries | None = None,
        *,
        agent: str = "",
        iteration: int = 0,
    ) -> None:
        self._retriever = retriever
        self._evidence_service = evidence_service
        self._knowledge = knowledge_repository
        self._evidence = evidence_repository
        self._papers = paper_repository
        self._bundles = bundle_repository
        self._graph_repository = graph_repository
        self._graph_queries = graph_queries
        self._agent = agent
        self._iteration = iteration
        self._calls: list[ToolCall] = []

    @property
    def calls(self) -> tuple[ToolCall, ...]:
        return tuple(self._calls)

    def for_agent(self, agent: str, iteration: int) -> ServiceToolbox:
        """A view that attributes its calls to one agent, sharing the same call log."""
        clone = ServiceToolbox(
            self._retriever,
            self._evidence_service,
            self._knowledge,
            self._evidence,
            self._papers,
            self._bundles,
            self._graph_repository,
            self._graph_queries,
            agent=agent,
            iteration=iteration,
        )
        clone._calls = self._calls
        return clone

    async def search_knowledge(
        self, query: str, *, kinds: tuple[str, ...] = (), limit: int = 10
    ) -> KnowledgeSearchResult:
        started = time.perf_counter()
        try:
            request = ResearchQuery(
                text=_clean_query(query),
                intent=QueryIntent.ANSWER,
                kinds=_parse_kinds(kinds),
                limit=_clamp(limit, MAX_LIMIT),
            )
            result = await self._retriever.retrieve(request)
        except ResearchAgentError as exc:
            self._record(ToolName.SEARCH_KNOWLEDGE, started, error=exc, query=query)
            return KnowledgeSearchResult(degraded=True)

        found = KnowledgeSearchResult(
            objects=tuple(hit.item for hit in result.hits),
            retrieved_by=result.retrieved_by,
            degraded=result.degraded,
        )
        self._record(
            ToolName.SEARCH_KNOWLEDGE, started, count=len(found.objects), query=query, limit=limit
        )
        return found

    async def retrieve_evidence(
        self, knowledge_object_ids: tuple[str, ...], *, limit: int = 20
    ) -> EvidenceSearchResult:
        started = time.perf_counter()
        wanted = set(knowledge_object_ids[:MAX_IDS])
        cap = _clamp(limit, MAX_LIMIT)

        records = []
        try:
            for paper_id in await self._knowledge.list_ids():
                stored = await self._evidence.get_paper(paper_id)
                if stored is None:
                    continue
                for record in stored.records:
                    if wanted & set(record.knowledge_object_ids):
                        records.append(record)
                    if len(records) >= cap:
                        break
                if len(records) >= cap:
                    break
        except ResearchAgentError as exc:
            self._record(ToolName.RETRIEVE_EVIDENCE, started, error=exc)
            return EvidenceSearchResult(degraded=True)

        self._record(ToolName.RETRIEVE_EVIDENCE, started, count=len(records))
        return EvidenceSearchResult(records=tuple(records))

    async def build_bundle(
        self, query: str, *, kinds: tuple[str, ...] = (), paper_ids: tuple[str, ...] = ()
    ) -> EvidenceBundle:
        """The only route to citable context.

        Raises rather than degrading: an agent that continues without a bundle would be
        reasoning over nothing, and every downstream citation traces to a bundle id.
        """
        started = time.perf_counter()
        request = ResearchQuery(
            text=_clean_query(query),
            intent=QueryIntent.ANSWER,
            kinds=_parse_kinds(kinds),
            paper_ids=tuple(paper_ids[:MAX_IDS]),
            limit=MAX_LIMIT,
        )
        try:
            bundle = await self._evidence_service.build_bundle(request)
            # Persisted, not merely returned. Findings cite bundle ids, and an id that
            # cannot be loaded back is an audit trail that dead-ends — the loop's later
            # stages and the audit builder both resolve citations through the repository.
            if self._bundles is not None:
                await self._bundles.save(bundle)
        except ResearchAgentError as exc:
            self._record(ToolName.BUILD_BUNDLE, started, error=exc, query=query)
            raise

        self._record(
            ToolName.BUILD_BUNDLE, started, count=len(bundle.knowledge_objects), query=query
        )
        return bundle

    async def search_graph(
        self, entity_name: str, *, depth: int = 1, limit: int = 25
    ) -> GraphSearchResult:
        started = time.perf_counter()
        if self._graph_repository is None or self._graph_queries is None:
            self._record(ToolName.SEARCH_GRAPH, started, count=0, entity=entity_name)
            return GraphSearchResult(available=False)

        try:
            versions = await self._graph_repository.versions()
            if not versions:
                self._record(ToolName.SEARCH_GRAPH, started, count=0, entity=entity_name)
                return GraphSearchResult(available=False)

            subgraph = await self._graph_repository.neighbours(
                _node_hint(entity_name),
                versions[0],
                depth=_clamp(depth, MAX_DEPTH),
                limit=_clamp(limit, MAX_LIMIT),
            )
            nodes = subgraph.nodes
            citations: tuple[str, ...] = ()
            if not nodes:
                # Fall back to a name lookup: the agent supplied a name, not a node id.
                matches = await self._graph_repository.find_nodes(
                    name=entity_name, version=versions[0], limit=_clamp(limit, MAX_LIMIT)
                )
                nodes = matches
            else:
                citations = tuple(
                    dict.fromkeys(
                        citation for edge in subgraph.edges for citation in edge.provenance.cite()
                    )
                )
        except ResearchAgentError as exc:
            self._record(ToolName.SEARCH_GRAPH, started, error=exc, entity=entity_name)
            return GraphSearchResult(available=False)

        self._record(ToolName.SEARCH_GRAPH, started, count=len(nodes), entity=entity_name)
        return GraphSearchResult(nodes=tuple(nodes), citations=citations)

    async def get_provenance(self, evidence_ids: tuple[str, ...]) -> tuple[str, ...]:
        started = time.perf_counter()
        addresses: list[str] = []
        try:
            for evidence_id in evidence_ids[:MAX_IDS]:
                record = await self._evidence.get(evidence_id)
                if record is not None:
                    addresses.append(record.evidence.location.describe())
        except ResearchAgentError as exc:
            self._record(ToolName.GET_PROVENANCE, started, error=exc)
            return ()

        self._record(ToolName.GET_PROVENANCE, started, count=len(addresses))
        return tuple(addresses)

    async def find_contradictions(
        self, paper_ids: tuple[str, ...] = ()
    ) -> tuple[Contradiction, ...]:
        started = time.perf_counter()
        try:
            wanted = tuple(paper_ids[:MAX_IDS]) or tuple(await self._knowledge.list_ids())
            objects: list[KnowledgeObject] = []
            for paper_id in wanted:
                stored = await self._knowledge.get(paper_id)
                if stored is not None:
                    objects.extend(stored.value.objects)
            from researchagent.services.evidence import ContradictionDetector

            found = ContradictionDetector().detect(tuple(objects))
        except ResearchAgentError as exc:
            self._record(ToolName.FIND_CONTRADICTIONS, started, error=exc)
            return ()

        self._record(ToolName.FIND_CONTRADICTIONS, started, count=len(found))
        return found

    async def get_paper_context(self, paper_id: str) -> PaperContext:
        started = time.perf_counter()
        try:
            stored = await self._knowledge.get(paper_id)
            record = await self._papers.get(paper_id)
        except ResearchAgentError as exc:
            self._record(ToolName.GET_PAPER_CONTEXT, started, error=exc, paper_id=paper_id)
            return PaperContext(paper_id=paper_id, found=False)

        if stored is None:
            self._record(ToolName.GET_PAPER_CONTEXT, started, count=0, paper_id=paper_id)
            return PaperContext(paper_id=paper_id, found=False)

        context = PaperContext(
            paper_id=paper_id,
            title=record.paper.title if record else "",
            year=record.paper.year if record else None,
            objects=stored.value.objects,
        )
        self._record(
            ToolName.GET_PAPER_CONTEXT, started, count=len(context.objects), paper_id=paper_id
        )
        return context

    def _record(
        self,
        tool: ToolName,
        started: float,
        *,
        count: int = 0,
        error: ResearchAgentError | None = None,
        **arguments: str | int | float | bool | None,
    ) -> None:
        call = ToolCall(
            tool=tool,
            arguments={key: value for key, value in arguments.items() if value is not None},
            result_count=count,
            latency_ms=(time.perf_counter() - started) * SECONDS_PER_MILLISECOND,
            succeeded=error is None,
            error=error.message if error else None,
            agent=self._agent,
            iteration=self._iteration,
        )
        self._calls.append(call)
        logger.debug(
            "tool_called",
            tool=tool.value,
            agent=self._agent,
            results=count,
            succeeded=call.succeeded,
        )


def _clamp(value: int, ceiling: int) -> int:
    """Bound whatever the model asked for. Never trust a model-supplied limit."""
    return max(1, min(value, ceiling))


def _clean_query(text: str) -> str:
    cleaned = " ".join(text.split())[:MAX_QUERY_CHARS]
    if not cleaned:
        raise ValueError("tool query must not be empty")
    return cleaned


def _parse_kinds(kinds: tuple[str, ...]) -> tuple[KnowledgeKind, ...]:
    """Unknown kinds are dropped, not guessed at.

    A model naming 'benchmark' means a filter nothing matches; silently dropping it gives
    the agent an unfiltered search, which is the safer failure.
    """
    parsed = []
    for kind in kinds:
        try:
            parsed.append(KnowledgeKind(kind.strip().lower()))
        except ValueError:
            logger.debug("tool_unknown_kind_ignored", kind=kind)
    return tuple(parsed)


def _node_hint(entity_name: str) -> str:
    """Graph nodes are addressed by id; an agent supplies a name.

    Returning the name unchanged lets `neighbours` miss, which the caller handles with a
    name lookup. Guessing an id format here would couple the toolbox to the id scheme.
    """
    return entity_name
