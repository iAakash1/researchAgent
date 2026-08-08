"""Cross-boundary contracts: workflow state and shared agent I/O.

Agent-local input/output models live in ``agents/<name>/schemas.py``; anything more than
one layer touches belongs here.
"""

from researchagent.schemas.workflow import (
    ResearchConstraints,
    ResearchState,
    RunStatus,
    StageFailure,
    StageRecord,
    StageStatus,
    WorkflowStage,
)

__all__ = [
    "ResearchConstraints",
    "ResearchState",
    "RunStatus",
    "StageFailure",
    "StageRecord",
    "StageStatus",
    "WorkflowStage",
]
