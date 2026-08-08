"""Research domain objects.

These are the nouns the whole system talks about. They know nothing about LLMs,
prompts, storage or HTTP.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field, field_validator


class QuestionPriority(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

    @property
    def rank(self) -> int:
        """Sort key: high first."""
        return {"high": 0, "medium": 1, "low": 2}[self.value]


class ResearchQuestion(BaseModel):
    """One answerable question the review must address."""

    id: str = Field(pattern=r"^RQ\d+$", description="Stable identifier, e.g. RQ1")
    question: str = Field(min_length=10)
    rationale: str = Field(min_length=10, description="Why this question matters")
    priority: QuestionPriority = QuestionPriority.MEDIUM
    keywords: list[str] = Field(default_factory=list, max_length=12)


class SearchStrategy(BaseModel):
    """How the literature will be found and filtered."""

    queries: list[str] = Field(min_length=1, max_length=20)
    inclusion_criteria: list[str] = Field(default_factory=list, max_length=10)
    exclusion_criteria: list[str] = Field(default_factory=list, max_length=10)
    year_from: int | None = Field(default=None, ge=1900, le=2100)


class ResearchPlan(BaseModel):
    """The Planner agent's deliverable: what to study and how to find it."""

    topic: str = Field(min_length=3)
    framing: str = Field(min_length=20, description="Scope and angle of the review")
    research_questions: list[ResearchQuestion] = Field(min_length=1, max_length=10)
    strategy: SearchStrategy
    expected_methods: list[str] = Field(default_factory=list, max_length=15)
    expected_datasets: list[str] = Field(default_factory=list, max_length=15)
    evaluation_metrics: list[str] = Field(default_factory=list, max_length=15)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("research_questions")
    @classmethod
    def _unique_ids(cls, questions: list[ResearchQuestion]) -> list[ResearchQuestion]:
        ids = [q.id for q in questions]
        if len(ids) != len(set(ids)):
            raise ValueError(f"research question ids must be unique, got {ids}")
        return questions

    def question(self, question_id: str) -> ResearchQuestion:
        for candidate in self.research_questions:
            if candidate.id == question_id:
                return candidate
        raise KeyError(question_id)
