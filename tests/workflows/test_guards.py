from __future__ import annotations

from pathlib import Path

from researchagent.models.paper import Paper, SourceName
from researchagent.models.research import (
    QuestionPriority,
    ResearchPlan,
    ResearchQuestion,
    SearchStrategy,
)
from researchagent.schemas.workflow import (
    ResearchState,
    RunStatus,
    StageFailure,
    WorkflowStage,
)
from researchagent.services.ranking import ScoredPaper
from researchagent.workflows.guards import (
    requires_candidates,
    requires_local_pdfs,
    requires_plan,
    run_not_failed,
)

GOAL = "Metastable failures in distributed systems"


def a_plan() -> ResearchPlan:
    return ResearchPlan(
        topic=GOAL,
        framing="A review of metastable failures and the conditions that trigger them.",
        research_questions=[
            ResearchQuestion(
                id="RQ1",
                question="What triggers metastable failures?",
                rationale="Triggers determine which mitigations apply at all.",
                priority=QuestionPriority.HIGH,
            )
        ],
        strategy=SearchStrategy(queries=["metastable failures"]),
    )


def candidate(local: Path | None = None) -> ScoredPaper:
    return ScoredPaper(
        paper=Paper(
            id="manual:01",
            title="Metastable failures",
            provider=SourceName.MANUAL,
            local_path=local,
        ),
        score=0.5,
    )


def state(**overrides: object) -> ResearchState:
    return ResearchState.model_validate({"goal": GOAL} | overrides)


class TestRunNotFailed:
    def test_allows_a_healthy_run(self) -> None:
        assert run_not_failed().check(state()).allowed is True

    def test_blocks_after_a_recorded_failure(self) -> None:
        """This is what stops a failed plan from being sent to the indexes."""
        failed = state(
            status=RunStatus.FAILED,
            failure=StageFailure(
                stage=WorkflowStage.PLANNING,
                agent="planner",
                code="output_parsing_error",
                message="bad json",
            ),
        )

        result = run_not_failed().check(failed)

        assert result.allowed is False
        assert result.reason is not None and "planning" in result.reason


class TestPrerequisiteGuards:
    def test_requires_plan(self) -> None:
        assert requires_plan().check(state()).allowed is False
        assert requires_plan().check(state(plan=a_plan())).allowed is True

    def test_requires_candidates(self) -> None:
        assert requires_candidates().check(state()).allowed is False
        assert requires_candidates().check(state(candidates=[candidate()])).allowed is True

    def test_requires_local_pdfs_rejects_metadata_only_candidates(self, tmp_path: Path) -> None:
        """Discovery can return papers it cannot fetch; parsing needs files on disk."""
        metadata_only = state(candidates=[candidate()])
        downloaded = state(candidates=[candidate(local=tmp_path / "a.pdf")])

        assert requires_local_pdfs().check(metadata_only).allowed is False
        assert requires_local_pdfs().check(downloaded).allowed is True

    def test_blocked_guards_name_what_is_missing(self) -> None:
        result = requires_local_pdfs().check(state(candidates=[candidate()]))

        assert result.missing == ("local_pdf",)
        assert result.reason is not None
