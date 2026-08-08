"""Conditional routing.

Routers read state and return a branch name; the graph maps that name to a node. Keeping
the decision here rather than inside a node is what lets the v0.8 reviewer loop send a
run back to the Planner without any stage knowing it happened.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from researchagent.core.logging import get_logger
from researchagent.schemas.workflow import ResearchState, RunStatus

logger = get_logger(__name__)

Branch = Literal["continue", "halt"]

CONTINUE: Branch = "continue"
HALT: Branch = "halt"


def halt_on_failure(state: ResearchState) -> Branch:
    """Stop the pipeline at the first failed stage.

    Stages record failures into state instead of raising, so without this every
    downstream stage would run against missing inputs and produce a second, misleading
    failure that buries the real one.
    """
    if state.status is RunStatus.FAILED or state.failure is not None:
        logger.info(
            "workflow_halted",
            run_id=state.run_id,
            stage=state.failure.stage.value if state.failure else None,
            reason=state.failure.code if state.failure else "failed",
        )
        return HALT
    return CONTINUE


def require(predicate: Callable[[ResearchState], bool]) -> Callable[[ResearchState], Branch]:
    """Continue only when the run has not failed *and* ``predicate`` holds.

    Used for stages with a precondition — retrieval needs candidates, parsing needs PDFs.
    """

    def router(state: ResearchState) -> Branch:
        if halt_on_failure(state) is HALT:
            return HALT
        return CONTINUE if predicate(state) else HALT

    return router
