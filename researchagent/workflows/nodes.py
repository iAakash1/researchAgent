"""Agent-to-node adaptation.

A LangGraph node is a function ``ResearchState -> partial update``. Agents know nothing
about state, so this adapter owns the translation in both directions plus the bookkeeping
every node needs: correlation context, stage records, and failure capture.

Failures are *recorded in state*, not raised. A run that dies with an exception loses its
partial results; a run that records a ``StageFailure`` stays inspectable, is persisted by
the checkpointer, and can be resumed.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel

from researchagent.agents.base import AgentContext, AgentResult, BaseAgent
from researchagent.core.constants import SECONDS_PER_MILLISECOND
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
from researchagent.workflows.guards import Guard, GuardResult

logger = get_logger(__name__)

StateUpdate = dict[str, Any]


class StageNode(ABC):
    """Shared stage bookkeeping: timing, history, failure capture, status transitions.

    Subclasses only say *what* the stage does. Everything a stage must do identically —
    so that the audit trail and failure semantics never drift between stages — lives here.
    """

    def __init__(
        self, stage: WorkflowStage, component: str, guards: list[Guard] | None = None
    ) -> None:
        self._stage = stage
        self._component = component
        self._guards = guards or []

    @property
    def stage(self) -> WorkflowStage:
        return self._stage

    @abstractmethod
    async def _execute(self, state: ResearchState) -> tuple[StateUpdate, float, int]:
        """Run the stage. Returns (state update, latency_ms, attempts)."""

    async def __call__(self, state: ResearchState) -> StateUpdate:
        blocked = self._first_unmet_guard(state)
        if blocked is not None:
            return self._blocked_update(blocked)

        try:
            payload, latency_ms, attempts = await self._execute(state)
        except ResearchAgentError as exc:
            return self._failure_update(_code_of(exc), exc.message)
        except Exception as exc:  # noqa: BLE001 - workflow boundary, see below
            # An unexpected exception must not take the whole run with it: the graph
            # would unwind and every earlier stage's work would be lost. Record it as a
            # stage failure so the run stays checkpointed and inspectable — but log the
            # traceback, because unlike a ResearchAgentError this is a bug.
            logger.bind(run_id=state.run_id, component=self._component).exception(
                "stage_crashed", stage=self._stage.value, error_type=type(exc).__name__
            )
            return self._failure_update("unexpected_error", f"{type(exc).__name__}: {exc}")

        update: StateUpdate = {
            "status": RunStatus.RUNNING,
            "current_stage": self._stage,
            "history": [
                StageRecord(
                    stage=self._stage,
                    agent=self._component,
                    status=StageStatus.OK,
                    latency_ms=latency_ms,
                    attempts=attempts,
                )
            ],
            "updated_at": datetime.now(UTC),
        }
        # The stage mapper wins: only it knows whether its stage is terminal, so it may
        # promote RUNNING to COMPLETED. Bookkeeping fields are its to overwrite too.
        update.update(payload)
        return update

    def _first_unmet_guard(self, state: ResearchState) -> GuardResult | None:
        """Prerequisites are checked before the stage body, never inside it."""
        for guard in self._guards:
            result = guard.check(state)
            if not result.allowed:
                return result
        return None

    def _blocked_update(self, result: GuardResult) -> StateUpdate:
        """A blocked stage is recorded, not silently skipped and not an exception.

        Whether blocking ends the run depends on why: a stage whose inputs are simply
        absent (no PDFs to parse) leaves a healthy run, while a stage blocked by an
        earlier failure keeps that failure.
        """
        logger.info(
            "stage_blocked",
            stage=self._stage.value,
            component=self._component,
            reason=result.reason,
            missing=list(result.missing),
        )
        return {
            "current_stage": self._stage,
            "history": [
                StageRecord(
                    stage=self._stage,
                    agent=self._component,
                    status=StageStatus.BLOCKED,
                    latency_ms=0.0,
                    note=result.reason,
                )
            ],
            "updated_at": datetime.now(UTC),
        }

    def _context(self, state: ResearchState) -> AgentContext:
        return AgentContext(
            run_id=state.run_id,
            session_id=state.session_id,
            metadata={"stage": self._stage.value, "iteration": state.iteration},
        )

    def _failure_update(self, code: str, message: str) -> StateUpdate:
        logger.error(
            "stage_failed",
            stage=self._stage.value,
            component=self._component,
            error_code=code,
            error=message,
        )
        return {
            "status": RunStatus.FAILED,
            "current_stage": self._stage,
            "failure": StageFailure(
                stage=self._stage,
                agent=self._component,
                code=code,
                message=message,
            ),
            "history": [
                StageRecord(
                    stage=self._stage,
                    agent=self._component,
                    status=StageStatus.FAILED,
                    latency_ms=0.0,
                )
            ],
            "updated_at": datetime.now(UTC),
        }


class AgentNode(StageNode):
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
        guards: list[Guard] | None = None,
    ) -> None:
        super().__init__(stage, agent.name, guards)
        self._agent = agent
        self._to_input = to_input
        self._to_update = to_update

    async def _execute(self, state: ResearchState) -> tuple[StateUpdate, float, int]:
        result = await self._agent.run(self._to_input(state), self._context(state))
        return self._to_update(result), result.latency_ms, result.attempts


class ServiceNode(StageNode):
    """Wraps a deterministic service call as a LangGraph node.

    Not every stage reasons. Discovery queries indexes, retrieval downloads files — those
    are services, and dressing them up as LLM agents would be dishonest and would drag an
    unused model binding and prompt along with them.
    """

    def __init__(
        self,
        stage: WorkflowStage,
        component: str,
        handler: Callable[[ResearchState], Awaitable[StateUpdate]],
        guards: list[Guard] | None = None,
    ) -> None:
        super().__init__(stage, component, guards)
        self._handler = handler

    async def _execute(self, state: ResearchState) -> tuple[StateUpdate, float, int]:
        started = time.perf_counter()
        update = await self._handler(state)
        return update, (time.perf_counter() - started) * SECONDS_PER_MILLISECOND, 1


def _code_of(exc: ResearchAgentError) -> str:
    """AgentExecutionError wraps the original cause; report that, not the wrapper."""
    if isinstance(exc, AgentExecutionError):
        return str(exc.context.get("cause", exc.code))
    return exc.code
