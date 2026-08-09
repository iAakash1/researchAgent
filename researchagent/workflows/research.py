"""The research workflow graph.

Current shape::

    START -> planning --(ok)--> discovery --(ok)--> document_intelligence
                  \\                      \\                    |
                   (failed) --> END         (failed) --> END    (ok)
                                                                 v
                                                        knowledge_extraction -> END

Guards, not defensive code inside stages, are what keep a stage from running on inputs
it cannot use.

Each subsequent version inserts a stage here and nowhere else: agents and services stay
unaware of what runs before or after them, which is the whole point of keeping control
flow in LangGraph.
"""

from __future__ import annotations

from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from researchagent.agents.base import AgentResult, BaseAgent
from researchagent.agents.planner.schemas import PlannerInput, PlannerOutput
from researchagent.core.logging import get_logger
from researchagent.schemas.validated import DocumentBatchResult
from researchagent.schemas.workflow import (
    DiscoveryReport,
    DocumentFailure,
    DocumentReport,
    KnowledgeReport,
    ResearchState,
    WorkflowStage,
)
from researchagent.services.discovery_service import DiscoveryService
from researchagent.services.document.pipeline import DocumentIntelligenceService
from researchagent.services.knowledge.pipeline import KnowledgeIntelligenceService
from researchagent.workflows.edges import CONTINUE, HALT, halt_on_failure
from researchagent.workflows.guards import (
    requires_candidates,
    requires_documents,
    requires_local_pdfs,
    requires_plan,
    run_not_failed,
)
from researchagent.workflows.nodes import AgentNode, ServiceNode, StateUpdate

logger = get_logger(__name__)


def planning_node(planner: BaseAgent[Any, Any]) -> AgentNode:
    def to_input(state: ResearchState) -> PlannerInput:
        return PlannerInput(
            goal=state.goal,
            constraints=state.constraints,
            feedback=state.feedback,
        )

    def to_update(result: AgentResult[Any]) -> StateUpdate:
        output: PlannerOutput = result.output
        return {"plan": output.plan}

    return AgentNode(
        planner,
        WorkflowStage.PLANNING,
        to_input=to_input,
        to_update=to_update,
        guards=[run_not_failed()],
    )


def discovery_node(discovery: DiscoveryService) -> ServiceNode:
    async def handler(state: ResearchState) -> StateUpdate:
        # `requires_plan` has already run; this narrows the type for the checker.
        assert state.plan is not None  # noqa: S101
        result = await discovery.discover(state.plan, run_id=state.run_id)
        return {
            "candidates": result.candidates,
            "discovery": DiscoveryReport(
                sources_queried=[report.source.value for report in result.reports],
                sources_failed=[source.value for source in result.sources_failed],
                papers_returned=result.total_returned,
                duplicates_removed=result.duplicates_removed,
                candidates=len(result.candidates),
            ),
        }

    return ServiceNode(
        WorkflowStage.DISCOVERY,
        "discovery_service",
        handler,
        guards=[run_not_failed(), requires_plan()],
    )


def document_intelligence_node(service: DocumentIntelligenceService) -> ServiceNode:
    async def handler(state: ResearchState) -> StateUpdate:
        papers = [candidate.paper for candidate in state.candidates]
        result: DocumentBatchResult = await service.process(papers, run_id=state.run_id)

        return {
            "documents": DocumentReport(
                attempted=len(result.outcomes),
                succeeded=result.succeeded,
                failed=result.failed,
                mean_confidence=result.mean_confidence,
                ready_for_extraction=tuple(
                    outcome.paper_id for outcome in result.outcomes if outcome.is_usable
                ),
                failures=tuple(
                    DocumentFailure(
                        paper_id=outcome.paper_id,
                        code=outcome.error_code or "unknown",
                        message=outcome.error_message or "",
                        remedy=outcome.remedy,
                    )
                    for outcome in result.outcomes
                    if not outcome.succeeded
                ),
            ),
        }

    return ServiceNode(
        WorkflowStage.DOCUMENT_INTELLIGENCE,
        "document_intelligence",
        handler,
        guards=[run_not_failed(), requires_candidates(), requires_local_pdfs()],
    )


def knowledge_extraction_node(service: KnowledgeIntelligenceService) -> ServiceNode:
    async def handler(state: ResearchState) -> StateUpdate:
        # `requires_documents` guarantees a validated document exists; the documents
        # themselves are loaded from the repository rather than carried in state.
        documents = await service.documents_for(state)
        result = await service.process(documents, run_id=state.run_id)

        kinds = sorted({kind for knowledge in result.knowledge for kind in knowledge.kinds_present})
        return {
            "knowledge": KnowledgeReport(
                documents_processed=len(result.outcomes),
                succeeded=result.succeeded,
                failed=result.failed,
                objects_extracted=result.total_objects,
                relations_built=sum(len(k.relations) for k in result.knowledge),
                rejected_ungrounded=sum(o.rejections.ungrounded for o in result.outcomes),
                rejected_invalid=sum(o.rejections.invalid for o in result.outcomes),
                grounding_rate=result.grounding_rate,
                kinds_present=tuple(kind.value for kind in kinds),
            ),
        }

    return ServiceNode(
        WorkflowStage.KNOWLEDGE_EXTRACTION,
        "knowledge_intelligence",
        handler,
        guards=[run_not_failed(), requires_documents()],
    )


def build_research_graph(
    *,
    planner: BaseAgent[Any, Any],
    discovery: DiscoveryService,
    documents: DocumentIntelligenceService,
    knowledge: KnowledgeIntelligenceService,
    checkpointer: BaseCheckpointSaver[str] | None = None,
) -> CompiledStateGraph[ResearchState, Any, ResearchState, ResearchState]:
    """Compile the workflow. Collaborators are injected, so tests compile it with fakes."""
    graph: StateGraph[ResearchState, Any, ResearchState, ResearchState] = StateGraph(ResearchState)

    graph.add_node(WorkflowStage.PLANNING.value, planning_node(planner))
    graph.add_node(WorkflowStage.DISCOVERY.value, discovery_node(discovery))
    graph.add_node(WorkflowStage.DOCUMENT_INTELLIGENCE.value, document_intelligence_node(documents))
    graph.add_node(WorkflowStage.KNOWLEDGE_EXTRACTION.value, knowledge_extraction_node(knowledge))

    graph.add_edge(START, WorkflowStage.PLANNING.value)
    graph.add_conditional_edges(
        WorkflowStage.PLANNING.value,
        halt_on_failure,
        {CONTINUE: WorkflowStage.DISCOVERY.value, HALT: END},
    )
    graph.add_conditional_edges(
        WorkflowStage.DISCOVERY.value,
        halt_on_failure,
        {CONTINUE: WorkflowStage.DOCUMENT_INTELLIGENCE.value, HALT: END},
    )
    graph.add_conditional_edges(
        WorkflowStage.DOCUMENT_INTELLIGENCE.value,
        halt_on_failure,
        {CONTINUE: WorkflowStage.KNOWLEDGE_EXTRACTION.value, HALT: END},
    )
    graph.add_edge(WorkflowStage.KNOWLEDGE_EXTRACTION.value, END)

    compiled = graph.compile(checkpointer=checkpointer)
    logger.info(
        "research_graph_compiled",
        stages=[stage.value for stage in WorkflowStage],
        checkpointing=checkpointer is not None,
    )
    return compiled
