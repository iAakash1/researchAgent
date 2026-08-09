"""Research queries — what the system is asked to find.

A :class:`ResearchQuery` is the single input shape every retrieval layer accepts. Layers
differ in what they return (knowledge, evidence, documents, bundles); they do not differ
in how they are asked.

The query is deliberately structural rather than free text. A reasoning engine that can
only ask "find me text like this" gets text back; one that can say "methods, from these
papers, above this confidence, mentioning these terms" gets evidence back. The filters
are the difference between retrieval and search.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, model_validator

from researchagent.models.knowledge import KnowledgeKind
from researchagent.models.research import ResearchQuestion


class QueryIntent(StrEnum):
    """Why the query is being asked. Retrieval layers use it to weight signals."""

    # Find everything about a topic; breadth matters more than precision.
    SURVEY = "survey"
    # Answer one specific research question; precision matters more than breadth.
    ANSWER = "answer"
    # Find agreement and disagreement across papers on one point.
    COMPARE = "compare"
    # Recover the provenance of something already known.
    TRACE = "trace"


class ResearchQuery(BaseModel):
    """A structured request for evidence."""

    model_config = {"frozen": True}

    text: str = Field(min_length=1, description="What is being asked, in words")
    intent: QueryIntent = QueryIntent.ANSWER

    # Filters. Empty means "no constraint", never "match nothing".
    kinds: tuple[KnowledgeKind, ...] = ()
    paper_ids: tuple[str, ...] = ()
    terms: tuple[str, ...] = Field(
        default=(), description="Domain terms to match, from the plan's keywords"
    )
    min_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    year_from: int | None = Field(default=None, ge=1500, le=2200)

    limit: int = Field(default=25, ge=1, le=500)
    # A question this query serves, when it came from a research plan. Carrying the id
    # rather than the text keeps the link to the plan checkable.
    question_id: str | None = None

    @model_validator(mode="after")
    def _terms_are_clean(self) -> ResearchQuery:
        if any(not term.strip() for term in self.terms):
            raise ValueError("query terms must not be blank")
        return self

    @classmethod
    def for_question(
        cls,
        question: ResearchQuestion,
        *,
        kinds: tuple[KnowledgeKind, ...] = (),
        paper_ids: tuple[str, ...] = (),
        limit: int = 25,
    ) -> ResearchQuery:
        """Build a query from a planner research question.

        The closing of the loop: the questions the Planner wrote in v0.2 become the
        retrieval requests that assemble evidence in v0.6, so a bundle can always name
        the question it exists to answer.
        """
        return cls(
            text=question.question,
            intent=QueryIntent.ANSWER,
            kinds=kinds,
            paper_ids=paper_ids,
            terms=tuple(keyword for keyword in question.keywords if keyword.strip()),
            limit=limit,
            question_id=question.id,
        )

    def matches_kind(self, kind: KnowledgeKind) -> bool:
        return not self.kinds or kind in self.kinds

    def matches_paper(self, paper_id: str) -> bool:
        return not self.paper_ids or paper_id in self.paper_ids

    def search_terms(self) -> tuple[str, ...]:
        """Query text and explicit terms together — what lexical matching runs against."""
        return (*self.text.split(), *self.terms)
