"""Workflow execution.

Owns the compiled graph and the LangGraph invocation details (thread ids, recursion
limits, stream modes) so no caller has to know them.

``run`` returns the final state even when the run failed, rather than raising. The
failure is data — it is checkpointed, inspectable and resumable — and the caller decides
whether that data becomes an HTTP error.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from uuid import uuid4

from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel

from researchagent.config.schemas import WorkflowConfig
from researchagent.core.exceptions import RunNotFoundError
from researchagent.core.logging import get_logger, log_context
from researchagent.schemas.workflow import (
    ResearchConstraints,
    ResearchState,
    RunStatus,
    StageFailure,
    WorkflowStage,
)

logger = get_logger(__name__)

ResearchGraph = CompiledStateGraph[ResearchState, Any, ResearchState, ResearchState]


class WorkflowUpdate(BaseModel):
    """One streamed step: which node ran and what it produced."""

    node: str
    status: RunStatus | None = None
    stage: WorkflowStage | None = None
    failure: StageFailure | None = None


class WorkflowRunner:
    def __init__(self, graph: ResearchGraph, config: WorkflowConfig) -> None:
        self._graph = graph
        self._config = config

    async def run(
        self,
        goal: str,
        *,
        constraints: ResearchConstraints | None = None,
        feedback: list[str] | None = None,
        run_id: str | None = None,
        session_id: str | None = None,
    ) -> ResearchState:
        """Execute the workflow to completion and return the final state."""
        state = self._initial_state(goal, constraints, feedback, run_id, session_id)

        with log_context(run_id=state.run_id):
            logger.info("workflow_started", goal=state.goal[:120])
            raw = await self._graph.ainvoke(state, config=self._invoke_config(state.run_id))
            final = _finalise(ResearchState.model_validate(raw))
            logger.info(
                "workflow_finished",
                status=final.status.value,
                stages=len(final.history),
                failed_at=final.failure.stage.value if final.failure else None,
            )
            return final

    async def stream(
        self,
        goal: str,
        *,
        constraints: ResearchConstraints | None = None,
        feedback: list[str] | None = None,
        run_id: str | None = None,
        session_id: str | None = None,
    ) -> AsyncIterator[WorkflowUpdate]:
        """Yield one update per completed node, as the graph progresses."""
        state = self._initial_state(goal, constraints, feedback, run_id, session_id)

        with log_context(run_id=state.run_id):
            logger.info("workflow_stream_started", goal=state.goal[:120])
            async for chunk in self._graph.astream(
                state, config=self._invoke_config(state.run_id), stream_mode="updates"
            ):
                for node, update in chunk.items():
                    yield _to_update(node, update)

    async def get_state(self, run_id: str) -> ResearchState:
        """Load a run from its checkpoint. Requires checkpointing to be enabled."""
        snapshot = await self._graph.aget_state(self._invoke_config(run_id))
        if not snapshot.values:
            raise RunNotFoundError("No checkpointed state for this run", run_id=run_id)

        state = ResearchState.model_validate(snapshot.values)
        # `snapshot.next` lists pending nodes; empty means the graph stopped. Settling
        # here as well keeps a reloaded run from reporting RUNNING when the same run was
        # returned as COMPLETED — the stored and returned truths must agree.
        return _finalise(state) if not snapshot.next else state

    @property
    def checkpointing_enabled(self) -> bool:
        return self._graph.checkpointer is not None

    def _initial_state(
        self,
        goal: str,
        constraints: ResearchConstraints | None,
        feedback: list[str] | None,
        run_id: str | None,
        session_id: str | None,
    ) -> ResearchState:
        return ResearchState(
            run_id=run_id or str(uuid4()),
            session_id=session_id,
            goal=goal,
            constraints=constraints or ResearchConstraints(),
            feedback=feedback or [],
            status=RunStatus.RUNNING,
        )

    def _invoke_config(self, run_id: str) -> RunnableConfig:
        # thread_id is LangGraph's checkpoint key; using run_id makes a run resumable and
        # retrievable by the same identifier the API hands back to the client.
        return RunnableConfig(
            configurable={"thread_id": run_id},
            recursion_limit=self._config.recursion_limit,
        )


def _finalise(state: ResearchState) -> ResearchState:
    """Settle the terminal status once the graph has stopped running.

    Individual stages must not have to know whether they are last — that would make the
    final stage's code change every time a stage is appended. When ``ainvoke`` returns,
    the graph is finished by definition, so a run still marked RUNNING with no recorded
    failure has completed.
    """
    if state.status is RunStatus.RUNNING and state.failure is None:
        return state.model_copy(update={"status": RunStatus.COMPLETED})
    return state


def _to_update(node: str, update: Any) -> WorkflowUpdate:
    if not isinstance(update, dict):
        return WorkflowUpdate(node=node)
    return WorkflowUpdate(
        node=node,
        status=update.get("status"),
        stage=update.get("current_stage"),
        failure=update.get("failure"),
    )
