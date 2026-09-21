"""The application use case that connects corpus building to audited reasoning."""

from __future__ import annotations

from researchagent.core.exceptions import RunNotFoundError
from researchagent.schemas.result import ResearchResult
from researchagent.schemas.workflow import ResearchConstraints
from researchagent.services.result import ResearchResultBuilder
from researchagent.workflows.reasoning_runner import ReasoningRunner
from researchagent.workflows.runner import WorkflowRunner


class ResearchRunService:
    def __init__(
        self,
        workflow: WorkflowRunner,
        reasoning: ReasoningRunner,
        results: ResearchResultBuilder,
    ) -> None:
        self._workflow = workflow
        self._reasoning = reasoning
        self._results = results
        self._completed: dict[str, ResearchResult] = {}

    async def run(
        self,
        goal: str,
        *,
        constraints: ResearchConstraints | None = None,
        feedback: list[str] | None = None,
        run_id: str | None = None,
        session_id: str | None = None,
    ) -> ResearchResult:
        state = await self._workflow.run(
            goal,
            constraints=constraints,
            feedback=feedback,
            run_id=run_id,
            session_id=session_id,
        )
        if state.succeeded:
            state = await self._reasoning.run(state)
        result = await self._results.build(state)
        self._completed[result.run_id] = result
        return result

    def get(self, run_id: str) -> ResearchResult:
        result = self._completed.get(run_id)
        if result is None:
            raise RunNotFoundError("No completed research result for this run", run_id=run_id)
        return result
