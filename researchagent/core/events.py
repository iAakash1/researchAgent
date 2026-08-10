"""In-process async event bus.

Producers (agents, services, workflow nodes) publish facts; consumers (metrics, SSE
streaming, persistence) subscribe. Producers never import consumers, so adding
observability never touches pipeline code.

Payloads are typed models, not dictionaries. A subscriber that has to guess which keys
an event carries is a subscriber that breaks silently when a producer is refactored —
and the telemetry that matters most is the telemetry nobody is watching until it is
needed.

A subscriber that raises is logged and skipped: observability must not break the run.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from enum import StrEnum
from uuid import uuid4

from pydantic import BaseModel, Field

from researchagent.core.logging import get_logger

logger = get_logger(__name__)


class EventType(StrEnum):
    AGENT_STARTED = "agent.started"
    AGENT_COMPLETED = "agent.completed"
    AGENT_FAILED = "agent.failed"
    AGENT_RETRIED = "agent.retried"

    LLM_CALL_COMPLETED = "llm.call.completed"

    WORKFLOW_STARTED = "workflow.started"
    WORKFLOW_COMPLETED = "workflow.completed"
    STAGE_BLOCKED = "workflow.stage.blocked"

    PAPER_DISCOVERED = "paper.discovered"
    PAPER_MERGED = "paper.merged"
    DISCOVERY_COMPLETED = "discovery.completed"

    DOCUMENT_LOADED = "document.loaded"
    DOCUMENT_PARSED = "document.parsed"
    DOCUMENT_READY = "document.ready"
    PARSING_FAILED = "document.parsing.failed"
    SECTIONS_DETECTED = "document.sections.detected"
    REFERENCES_EXTRACTED = "document.references.extracted"

    KNOWLEDGE_EXTRACTED = "knowledge.extracted"
    EVIDENCE_INDEXED = "evidence.indexed"
    BUNDLE_CREATED = "bundle.created"
    BUNDLE_MERGED = "bundle.merged"
    RETRIEVAL_PERFORMED = "retrieval.performed"
    INDEX_BUILT = "index.built"
    GRAPH_BUILT = "graph.built"
    GRAPH_EDGE_REJECTED = "graph.edge.rejected"

    # v0.9 agentic loop. Tool calls and iteration boundaries are events rather than log
    # lines because they are what an audit of a conclusion needs to replay.
    TOOL_CALLED = "tool.called"
    TOOL_COMPLETED = "tool.completed"
    RESEARCH_ITERATION_STARTED = "research.iteration.started"
    HYPOTHESIS_CREATED = "research.hypothesis.created"
    FINDING_CREATED = "research.finding.created"
    FINDING_VERIFIED = "research.finding.verified"
    FINDING_REJECTED = "research.finding.rejected"
    RESEARCH_TERMINATED = "research.terminated"
    CONTRADICTION_DETECTED = "contradiction.detected"
    KNOWLEDGE_REJECTED = "knowledge.rejected"

    VALIDATION_PASSED = "validation.passed"
    VALIDATION_FAILED = "validation.failed"
    EVIDENCE_GENERATED = "evidence.generated"


class EventPayload(BaseModel):
    """Base for every event body. Subclass rather than reaching for a dict.

    ``extra="forbid"`` is load-bearing: without it a dict payload validates into an empty
    base instance and every field is silently dropped, which is worse than the untyped
    dict it replaced.
    """

    model_config = {"frozen": True, "extra": "forbid"}


class AgentPayload(EventPayload):
    agent: str
    latency_ms: float | None = None
    attempts: int | None = None
    error: str | None = None
    code: str | None = None


class ReasoningPayload(EventPayload):
    """One step of the agentic loop.

    Carries ids and counts, never prompts or model output: an event stream that quoted
    what an agent was told would leak whatever the corpus contains into the log.
    """

    agent: str
    iteration: int = 0
    question_id: str | None = None
    finding_id: str | None = None
    hypothesis_id: str | None = None
    verdict: str | None = None
    bundles: int | None = None
    findings: int | None = None
    detail: str | None = None


class ToolCallPayload(EventPayload):
    """A tool invocation. Arguments are summarised, never echoed verbatim."""

    agent: str
    tool: str
    iteration: int = 0
    result_count: int = 0
    latency_ms: float = 0.0
    succeeded: bool = True
    error_code: str | None = None


class LLMCallPayload(EventPayload):
    alias: str
    model: str
    latency_ms: float
    prompt_tokens: int = 0
    completion_tokens: int = 0


class StagePayload(EventPayload):
    stage: str
    reason: str | None = None
    missing: tuple[str, ...] = ()


class PaperPayload(EventPayload):
    paper_id: str
    provider: str
    title: str | None = None
    merged_from: tuple[str, ...] = ()


class DiscoveryPayload(EventPayload):
    sources_queried: tuple[str, ...] = ()
    sources_failed: tuple[str, ...] = ()
    papers_returned: int = 0
    duplicates_removed: int = 0
    candidates: int = 0


class DocumentPayload(EventPayload):
    paper_id: str
    pages: int | None = None
    sections: int | None = None
    references: int | None = None
    figures: int | None = None
    tables: int | None = None
    citations: int | None = None
    duration_ms: float | None = None
    error_code: str | None = None
    error_message: str | None = None


class KnowledgePayload(EventPayload):
    paper_id: str
    objects: int = 0
    relations: int = 0
    evidence: int = 0
    kinds: tuple[str, ...] = ()
    rejected: int = 0


class BundlePayload(EventPayload):
    bundle_id: str
    question_id: str | None = None
    knowledge_objects: int = 0
    evidence_items: int = 0
    papers: int = 0
    contradictions: int = 0
    confidence: float = 0.0


class RetrievalPayload(EventPayload):
    layer: str
    retrieved_by: str
    query: str
    hits: int = 0
    considered: int = 0
    latency_ms: float = 0.0


class IndexPayload(EventPayload):
    index_version: str
    model_fingerprint: str
    objects: int = 0
    embedding_ms: float = 0.0


class GraphPayload(EventPayload):
    graph_version: str
    nodes: int = 0
    edges: int = 0
    rejected_edges: int = 0
    contradictions: int = 0
    provenance_coverage: float = 0.0


class ValidationPayload(EventPayload):
    validator: str
    subject_id: str
    subject_type: str
    success: bool
    confidence: float
    issue_codes: tuple[str, ...] = ()


class EvidencePayload(EventPayload):
    document_id: str
    produced_by: str
    count: int
    kinds: tuple[str, ...] = ()


class Event(BaseModel):
    """Immutable record of something that happened."""

    model_config = {"frozen": True}

    type: EventType
    payload: EventPayload = Field(default_factory=EventPayload)
    run_id: str | None = None
    source: str | None = None
    event_id: str = Field(default_factory=lambda: str(uuid4()))
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


Subscriber = Callable[[Event], Awaitable[None]]


class EventBus:
    """Fan-out bus with per-type and wildcard subscribers."""

    def __init__(self) -> None:
        self._subscribers: dict[EventType | None, list[Subscriber]] = defaultdict(list)

    def subscribe(self, event_type: EventType | None, handler: Subscriber) -> Callable[[], None]:
        """Register ``handler``; ``event_type=None`` receives every event.

        Returns an unsubscribe callable.
        """
        self._subscribers[event_type].append(handler)

        def unsubscribe() -> None:
            handlers = self._subscribers.get(event_type, [])
            if handler in handlers:
                handlers.remove(handler)

        return unsubscribe

    async def publish(self, event: Event) -> None:
        """Deliver to all matching subscribers concurrently, isolating failures."""
        handlers = [*self._subscribers.get(event.type, []), *self._subscribers.get(None, [])]
        if not handlers:
            return

        results = await asyncio.gather(
            *(handler(event) for handler in handlers), return_exceptions=True
        )
        for handler, result in zip(handlers, results, strict=True):
            if isinstance(result, BaseException):
                logger.error(
                    "event_subscriber_failed",
                    event_type=event.type,
                    handler=getattr(handler, "__qualname__", repr(handler)),
                    error=str(result),
                    error_type=type(result).__name__,
                )

    async def emit(
        self,
        event_type: EventType,
        payload: EventPayload,
        *,
        run_id: str | None = None,
        source: str | None = None,
    ) -> None:
        """Convenience for the common publish; keeps call sites to one line."""
        await self.publish(Event(type=event_type, payload=payload, run_id=run_id, source=source))

    def subscriber_count(self, event_type: EventType | None = None) -> int:
        return len(self._subscribers.get(event_type, []))
