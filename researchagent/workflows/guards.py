"""Workflow guards: prerequisites checked before a stage runs.

A stage should not have to defend itself against being called in the wrong order. The
guard states what a stage needs; the graph refuses to enter it otherwise and records a
blocked stage in the audit trail.

This is where "verification refuses to run if parsing failed" is enforced — once, in
declarative form, rather than as defensive code scattered through every stage body.
"""

from __future__ import annotations

from collections.abc import Callable

from pydantic import BaseModel, Field

from researchagent.core.logging import get_logger
from researchagent.schemas.workflow import ResearchState, RunStatus

logger = get_logger(__name__)


class GuardResult(BaseModel):
    """Whether a stage may run, and what is missing if not."""

    model_config = {"frozen": True}

    allowed: bool
    reason: str | None = None
    missing: tuple[str, ...] = ()

    @classmethod
    def allow(cls) -> GuardResult:
        return cls(allowed=True)

    @classmethod
    def block(cls, reason: str, *missing: str) -> GuardResult:
        return cls(allowed=False, reason=reason, missing=tuple(missing))


class Guard(BaseModel):
    """A named prerequisite over the workflow state."""

    model_config = {"frozen": True, "arbitrary_types_allowed": True}

    name: str = Field(min_length=1)
    description: str = ""
    predicate: Callable[[ResearchState], GuardResult]

    def check(self, state: ResearchState) -> GuardResult:
        result = self.predicate(state)
        if not result.allowed:
            logger.info(
                "stage_prerequisite_unmet",
                guard=self.name,
                run_id=state.run_id,
                reason=result.reason,
                missing=list(result.missing),
            )
        return result


def run_not_failed() -> Guard:
    """Universal precondition: never run a stage inside an already-failed run."""

    def predicate(state: ResearchState) -> GuardResult:
        if state.status is RunStatus.FAILED or state.failure is not None:
            stage = state.failure.stage.value if state.failure else "unknown"
            return GuardResult.block(f"run already failed at stage {stage}", "healthy_run")
        return GuardResult.allow()

    return Guard(
        name="run_not_failed", description="The run has no recorded failure", predicate=predicate
    )


def requires_plan() -> Guard:
    def predicate(state: ResearchState) -> GuardResult:
        if state.plan is None:
            return GuardResult.block("no research plan in state", "plan")
        return GuardResult.allow()

    return Guard(name="requires_plan", description="Planning produced a plan", predicate=predicate)


def requires_candidates() -> Guard:
    def predicate(state: ResearchState) -> GuardResult:
        if not state.candidates:
            return GuardResult.block("discovery returned no candidates", "candidates")
        return GuardResult.allow()

    return Guard(
        name="requires_candidates",
        description="Discovery produced at least one paper",
        predicate=predicate,
    )


def requires_local_pdfs() -> Guard:
    """Document intelligence needs files on disk, not merely metadata."""

    def predicate(state: ResearchState) -> GuardResult:
        available = [c for c in state.candidates if c.paper.local_path is not None]
        if not available:
            return GuardResult.block("no candidate has a local PDF to parse", "local_pdf")
        return GuardResult.allow()

    return Guard(
        name="requires_local_pdfs",
        description="At least one candidate has a downloaded PDF",
        predicate=predicate,
    )


def all_of(*guards: Guard) -> list[Guard]:
    """Compose guards; the first block wins so the reason stays specific."""
    return list(guards)
