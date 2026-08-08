"""Planner contracts.

Two schema families, deliberately kept apart:

* ``PlannerInput`` / ``PlannerOutput`` — the agent's contract with the workflow.
* ``*Draft`` — what the *model* is asked to produce. These are flatter and looser than
  the domain models: an 8B model reliably fills a shallow schema and reliably fails a
  deep one. The agent maps drafts onto ``ResearchPlan``, applying ids, ordering and
  limits deterministically instead of trusting the model with them.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from researchagent.models.research import QuestionPriority, ResearchPlan
from researchagent.schemas.workflow import ResearchConstraints


class PlannerOptions(BaseModel):
    """Tunables from ``config/agents.yaml`` under ``planner.options``."""

    min_research_questions: int = Field(default=3, ge=1, le=10)
    max_research_questions: int = Field(default=5, ge=1, le=10)
    max_queries: int = Field(default=8, ge=1, le=20)
    max_keywords_per_question: int = Field(default=6, ge=1, le=12)

    @model_validator(mode="after")
    def _check_bounds(self) -> PlannerOptions:
        if self.min_research_questions > self.max_research_questions:
            raise ValueError(
                "min_research_questions must not exceed max_research_questions "
                f"({self.min_research_questions} > {self.max_research_questions})"
            )
        return self


class PlannerInput(BaseModel):
    goal: str = Field(min_length=8, description="The user's research goal, verbatim")
    constraints: ResearchConstraints = Field(default_factory=ResearchConstraints)
    # Reviewer critique from a previous iteration; drives re-planning in the review loop.
    feedback: list[str] = Field(default_factory=list, max_length=20)


class PlannerOutput(BaseModel):
    plan: ResearchPlan


class QuestionDraft(BaseModel):
    """One research question as produced by the model."""

    question: str
    rationale: str
    priority: QuestionPriority = QuestionPriority.MEDIUM
    keywords: list[str] = Field(default_factory=list)


class FramingDraft(BaseModel):
    """Phase 1 output: what exactly are we studying, and what must we answer?"""

    topic: str
    framing: str
    questions: list[QuestionDraft] = Field(default_factory=list)


class StrategyDraft(BaseModel):
    """Phase 2 output: how do we find and filter the literature?"""

    queries: list[str] = Field(default_factory=list)
    inclusion_criteria: list[str] = Field(default_factory=list)
    exclusion_criteria: list[str] = Field(default_factory=list)
    expected_methods: list[str] = Field(default_factory=list)
    expected_datasets: list[str] = Field(default_factory=list)
    evaluation_metrics: list[str] = Field(default_factory=list)
