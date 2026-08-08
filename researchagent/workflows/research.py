"""The research workflow graph.

Current shape::

    START -> planning -> END

Each subsequent version inserts a stage here and nowhere else: agents stay unaware of
what runs before or after them, which is the whole point of keeping control flow in
LangGraph.
"""

from __future__ import annotations

from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from researchagent.agents.base import AgentResult, BaseAgent
from researchagent.agents.planner.schemas import PlannerInput, PlannerOutput
from researchagent.core.logging import get_logger
from researchagent.schemas.workflow import ResearchState, RunStatus, WorkflowStage
from researchagent.workflows.nodes import AgentNode, StateUpdate

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
        # Terminal for now: planning is the last stage in the graph. The status moves to
        # RUNNING (set by AgentNode) then COMPLETED here until a successor stage exists.
        return {"plan": output.plan, "status": RunStatus.COMPLETED}

    return AgentNode(planner, WorkflowStage.PLANNING, to_input=to_input, to_update=to_update)


def build_research_graph(
    *,
    planner: BaseAgent[Any, Any],
    checkpointer: BaseCheckpointSaver[str] | None = None,
) -> CompiledStateGraph[ResearchState, Any, ResearchState, ResearchState]:
    """Compile the workflow. Agents are injected, so tests can compile it with fakes."""
    graph: StateGraph[ResearchState, Any, ResearchState, ResearchState] = StateGraph(ResearchState)

    graph.add_node(WorkflowStage.PLANNING.value, planning_node(planner))
    graph.add_edge(START, WorkflowStage.PLANNING.value)
    graph.add_edge(WorkflowStage.PLANNING.value, END)

    compiled = graph.compile(checkpointer=checkpointer)
    logger.info(
        "research_graph_compiled",
        stages=[WorkflowStage.PLANNING.value],
        checkpointing=checkpointer is not None,
    )
    return compiled
