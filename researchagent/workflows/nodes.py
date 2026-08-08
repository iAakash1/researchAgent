"""Agent-to-node adaptation.

A LangGraph node is a function ``ResearchState -> partial update``. Agents know nothing
about state, so this adapter owns the translation in both directions plus the bookkeeping
every node needs: correlation context, stage records, and failure capture.

Failures are *recorded in state*, not raised. A run that dies with an exception loses its
partial results; a run that records a ``StageFailure`` stays inspectable, is persisted by
the checkpointer, and can be resumed.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel

from researchagent.agents.base import AgentContext, AgentResult, BaseAgent
from researchagent.core.exceptions import AgentExecutionError, ResearchAgentError
from researchagent.core.logging import get_logger
from researchagent.schemas.workflow import (
    ResearchState,
    RunStatus,
    StageFailure,
    StageRecord,
    StageStatus,
    WorkflowStage,
)

logger = get_logger(__name__)

StateUpdate = dict[str, Any]


class AgentNode:
    """Wraps one agent as a LangGraph node.

    ``to_input`` projects the state onto the agent's input contract; ``to_update``
    projects the agent's output back onto state fields. Neither side imports the other.
    """

    def __init__(
        self,
        agent: BaseAgent[Any, Any],
        stage: WorkflowStage,
        *,
        to_input: Callable[[ResearchState], BaseModel],
        to_update: Callable[[AgentResult[Any]], StateUpdate],
    ) -> None:
        self._agent = agent
        self._stage = stage
        self._to_input = to_input
        self._to_update = to_update

    @property
    def stage(self) -> WorkflowStage:
        return self._stage

    async def __call__(self, state: ResearchState) -> StateUpdate:
        context = AgentContext(
            run_id=state.run_id,
            session_id=state.session_id,
            metadata={"stage": self._stage.value, "iteration": state.iteration},
        )

        try:
            result = await self._agent.run(self._to_input(state), context)
        except ResearchAgentError as exc:
            return self._failure_update(_code_of(exc), exc.message)
        except Exception as exc:  # noqa: BLE001 - workflow boundary, see below
            # An unexpected exception must not take the whole run with it: the graph
            # would unwind and every earlier stage's work would be lost. Record it as a
            # stage failure so the run stays checkpointed and inspectable — but log the
            # traceback, because unlike a ResearchAgentError this is a bug.
            self._logger_for(state).exception(
                "stage_crashed", stage=self._stage.value, error_type=type(exc).__name__
            )
            return self._failure_update("unexpected_error", f"{type(exc).__name__}: {exc}")

        update: StateUpdate = {
            "status": RunStatus.RUNNING,
            "current_stage": self._stage,
            "history": [
                StageRecord(
                    stage=self._stage,
                    agent=self._agent.name,
                    status=StageStatus.OK,
                    latency_ms=result.latency_ms,
                    attempts=result.attempts,
                )
            ],
            "updated_at": datetime.now(UTC),
        }
        # The stage mapper wins: only it knows whether its stage is terminal, so it may
        # promote RUNNING to COMPLETED. Bookkeeping fields are its to overwrite too.
        update.update(self._to_update(result))
        return update

    def _logger_for(self, state: ResearchState) -> Any:
        return logger.bind(run_id=state.run_id, agent=self._agent.name)

    def _failure_update(self, code: str, message: str) -> StateUpdate:
        logger.error(
            "stage_failed",
            stage=self._stage.value,
            agent=self._agent.name,
            error_code=code,
            error=message,
        )
        return {
            "status": RunStatus.FAILED,
            "current_stage": self._stage,
            "failure": StageFailure(
                stage=self._stage,
                agent=self._agent.name,
                code=code,
                message=message,
            ),
            "history": [
                StageRecord(
                    stage=self._stage,
                    agent=self._agent.name,
                    status=StageStatus.FAILED,
                    latency_ms=0.0,
                )
            ],
            "updated_at": datetime.now(UTC),
        }


def _code_of(exc: ResearchAgentError) -> str:
    """AgentExecutionError wraps the original cause; report that, not the wrapper."""
    if isinstance(exc, AgentExecutionError):
        return str(exc.context.get("cause", exc.code))
    return exc.code
