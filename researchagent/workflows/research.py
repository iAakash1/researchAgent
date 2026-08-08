"""The research workflow graph.

Current shape::

    START -> planning --(ok)--> discovery -> END
                  \\
                   (failed) --> END

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
from researchagent.schemas.workflow import (
    DiscoveryReport,
    ResearchState,
    RunStatus,
    WorkflowStage,
)
from researchagent.services.discovery_service import DiscoveryService
from researchagent.workflows.edges import CONTINUE, HALT, halt_on_failure
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

    return AgentNode(planner, WorkflowStage.PLANNING, to_input=to_input, to_update=to_update)


def discovery_node(discovery: DiscoveryService) -> ServiceNode:
    async def handler(state: ResearchState) -> StateUpdate:
        if state.plan is None:
            # Unreachable through the graph (planning gates this stage), but an explicit
            # guard beats an AttributeError if the graph is ever rewired.
            raise ValueError("discovery requires a research plan")

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
            # Terminal stage for now.
            "status": RunStatus.COMPLETED,
        }

    return ServiceNode(WorkflowStage.DISCOVERY, "discovery_service", handler)


def build_research_graph(
    *,
    planner: BaseAgent[Any, Any],
    discovery: DiscoveryService,
    checkpointer: BaseCheckpointSaver[str] | None = None,
) -> CompiledStateGraph[ResearchState, Any, ResearchState, ResearchState]:
    """Compile the workflow. Collaborators are injected, so tests compile it with fakes."""
    graph: StateGraph[ResearchState, Any, ResearchState, ResearchState] = StateGraph(ResearchState)

    graph.add_node(WorkflowStage.PLANNING.value, planning_node(planner))
    graph.add_node(WorkflowStage.DISCOVERY.value, discovery_node(discovery))

    graph.add_edge(START, WorkflowStage.PLANNING.value)
    graph.add_conditional_edges(
        WorkflowStage.PLANNING.value,
        halt_on_failure,
        {CONTINUE: WorkflowStage.DISCOVERY.value, HALT: END},
    )
    graph.add_edge(WorkflowStage.DISCOVERY.value, END)

    compiled = graph.compile(checkpointer=checkpointer)
    logger.info(
        "research_graph_compiled",
        stages=[WorkflowStage.PLANNING.value, WorkflowStage.DISCOVERY.value],
        checkpointing=checkpointer is not None,
    )
    return compiled
