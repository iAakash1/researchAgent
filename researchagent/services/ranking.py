"""Relevance ranking.

Deliberately a lexical heuristic, and deliberately behind an interface: v0.5 replaces the
scoring with embedding similarity and a cross-encoder reranker. What must survive that
swap is the *shape* — a ``PaperScorer`` producing a score plus its per-signal breakdown.

The breakdown is not decoration. A ranking you cannot explain is a ranking you cannot
debug, and the v0.8 reviewer needs to say why a paper was included.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field

from researchagent.config.schemas import RankingConfig
from researchagent.core.logging import get_logger
from researchagent.models.paper import Paper, normalise_title
from researchagent.models.research import ResearchPlan

logger = get_logger(__name__)

_STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for",
        "from", "how", "in", "into", "is", "it", "its", "of", "on", "or", "our", "that",
        "the", "their", "there", "these", "this", "those", "to", "via", "we", "what",
        "which", "why", "with",
    }
)  # fmt: skip

# Method words occur across nearly every scientific field. They may explain how a paper
# works, but they cannot establish what the paper is about.
_GENERIC_RESEARCH_TERMS = frozenset(
    {
        "analysis",
        "application",
        "approach",
        "based",
        "classification",
        "classify",
        "deep",
        "detect",
        "detection",
        "identify",
        "identification",
        "learning",
        "machine",
        "method",
        "model",
        "neural",
        "predict",
        "prediction",
        "recognition",
        "study",
        "survey",
        "system",
        "using",
    }
)


class RelevanceDecision(StrEnum):
    DIRECT = "directly_relevant"
    RELATED = "related_but_broad"
    IRRELEVANT = "irrelevant"

    @property
    def rank(self) -> int:
        return {
            RelevanceDecision.DIRECT: 0,
            RelevanceDecision.RELATED: 1,
            RelevanceDecision.IRRELEVANT: 2,
        }[self]


class ScoredPaper(BaseModel):
    paper: Paper
    score: float = Field(ge=0.0, le=1.0)
    # signal name -> contribution, always summing to `score`.
    signals: dict[str, float] = Field(default_factory=dict)
    relevance_score: float = Field(default=1.0, ge=0.0, le=1.0)
    relevance_decision: RelevanceDecision = RelevanceDecision.DIRECT
    relevance_signals: dict[str, float] = Field(default_factory=dict)

    @property
    def id(self) -> str:
        return self.paper.id


class PaperScorer(ABC):
    """Port for relevance scoring. v0.5 adds an embedding-based implementation."""

    @abstractmethod
    def score(
        self, paper: Paper, plan: ResearchPlan, *, research_goal: str | None = None
    ) -> ScoredPaper: ...

    def rank(
        self,
        papers: list[Paper],
        plan: ResearchPlan,
        *,
        research_goal: str | None = None,
        limit: int | None = None,
    ) -> list[ScoredPaper]:
        scored = [self.score(paper, plan, research_goal=research_goal) for paper in papers]
        # Ties broken by citation count then year: reproducible ordering matters for
        # experiment comparability.
        scored.sort(
            key=lambda item: (
                item.relevance_decision.rank,
                -item.relevance_score,
                -item.score,
                -(item.paper.citation_count or 0),
                -(item.paper.year or 0),
                item.paper.title,
            )
        )
        return scored[:limit] if limit is not None else scored


class HeuristicScorer(PaperScorer):
    """Lexical relevance: term overlap, recency and citation impact."""

    def __init__(self, config: RankingConfig | None = None) -> None:
        self._config = config or RankingConfig()

    def score(
        self, paper: Paper, plan: ResearchPlan, *, research_goal: str | None = None
    ) -> ScoredPaper:
        weights = self._config.weights
        query_terms = _plan_terms(plan)
        plan_keywords = {
            kw.lower() for question in plan.research_questions for kw in question.keywords
        }

        signals = {
            "title_match": weights.title_match * _overlap(_tokenise(paper.title), query_terms),
            "abstract_match": weights.abstract_match
            * _overlap(_tokenise(paper.abstract or ""), query_terms),
            "keyword_overlap": weights.keyword_overlap
            * _overlap({kw.lower() for kw in paper.keywords}, plan_keywords),
            "recency": weights.recency * self._recency(paper),
            "citations": weights.citations * self._citations(paper),
        }

        total_weight = weights.total()
        if total_weight <= 0:
            score = 0.0
            normalised: dict[str, float] = {}
        else:
            # Round the contributions first, then sum them, so `signals` always adds up to
            # `score` exactly — an explanation that does not reconcile is worse than none.
            normalised = {name: round(value / total_weight, 6) for name, value in signals.items()}
            score = min(round(sum(normalised.values()), 6), 1.0)

        relevance_score, decision, relevance_signals = self._relevance(
            paper, plan, research_goal or plan.topic
        )
        return ScoredPaper(
            paper=paper,
            score=score,
            signals=normalised,
            relevance_score=relevance_score,
            relevance_decision=decision,
            relevance_signals=relevance_signals,
        )

    def _relevance(
        self, paper: Paper, plan: ResearchPlan, research_goal: str
    ) -> tuple[float, RelevanceDecision, dict[str, float]]:
        goal_terms = _distinctive_tokens(research_goal)
        if not goal_terms:
            goal_terms = _tokenise(research_goal)

        title_terms = _tokenise(paper.title)
        abstract_terms = _tokenise(paper.abstract or "")
        keyword_terms = _tokenise(" ".join(paper.keywords))
        all_terms = title_terms | abstract_terms | keyword_terms

        title_alignment = _overlap(title_terms, goal_terms)
        abstract_alignment = _overlap(abstract_terms, goal_terms)
        keyword_alignment = _overlap(keyword_terms, goal_terms)
        goal_alignment = _overlap(all_terms, goal_terms)

        concepts = _plan_concepts(plan)
        concept_alignment = max((_overlap(all_terms, concept) for concept in concepts), default=0.0)
        exact_concept = any(len(concept) >= 2 and concept <= all_terms for concept in concepts)
        relevance_score = round(
            max(
                title_alignment,
                abstract_alignment * 0.9,
                keyword_alignment * 0.95,
                goal_alignment * 0.9,
                concept_alignment * 0.9,
            ),
            6,
        )

        matched_goal_terms = len(all_terms & goal_terms)
        required_matches = min(self._config.min_distinctive_matches, len(goal_terms))
        directly_relevant = (
            matched_goal_terms >= required_matches
            and relevance_score >= self._config.direct_relevance_threshold
        ) or exact_concept
        if directly_relevant:
            decision = RelevanceDecision.DIRECT
        elif (
            matched_goal_terms > 0 and relevance_score >= self._config.related_relevance_threshold
        ) or concept_alignment >= self._config.related_relevance_threshold:
            decision = RelevanceDecision.RELATED
        else:
            decision = RelevanceDecision.IRRELEVANT

        return (
            relevance_score,
            decision,
            {
                "goal_title_alignment": round(title_alignment, 6),
                "goal_abstract_alignment": round(abstract_alignment, 6),
                "goal_keyword_alignment": round(keyword_alignment, 6),
                "goal_alignment": round(goal_alignment, 6),
                "concept_alignment": round(concept_alignment, 6),
            },
        )

    def _recency(self, paper: Paper) -> float:
        if paper.year is None:
            # Unknown year is not evidence of being old; score it neutrally rather than
            # burying manual papers and preprints that omit a date.
            return 0.5
        age = datetime.now(UTC).year - paper.year
        if age <= 0:
            return 1.0
        return math.exp(-age / self._config.recency_half_life_years)

    def _citations(self, paper: Paper) -> float:
        if paper.citation_count is None:
            # Providers that do not report citations (arXiv, manual) must not be
            # penalised against those that do.
            return 0.5
        if paper.citation_count <= 0:
            return 0.0
        return min(
            math.log1p(paper.citation_count) / math.log1p(self._config.citation_saturation), 1.0
        )


def _plan_terms(plan: ResearchPlan) -> set[str]:
    """Everything the plan says it is looking for, as a bag of terms."""
    parts = [plan.topic, *plan.strategy.queries]
    for question in plan.research_questions:
        parts.append(question.question)
        parts.extend(question.keywords)
    return _tokenise(" ".join(parts))


def _plan_concepts(plan: ResearchPlan) -> tuple[set[str], ...]:
    concepts = []
    for phrase in plan.strategy.queries:
        terms = _distinctive_tokens(phrase)
        if terms and terms not in concepts:
            concepts.append(terms)
    return tuple(concepts)


def _distinctive_tokens(text: str) -> set[str]:
    return _tokenise(text) - _GENERIC_RESEARCH_TERMS


def _tokenise(text: str) -> set[str]:
    return {
        _stem(token)
        for token in normalise_title(text).split()
        if len(token) > 2 and token not in _STOPWORDS
    }


def _stem(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"):
        return f"{token[:-3]}y"
    if len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _overlap(candidate: set[str], reference: set[str]) -> float:
    """Fraction of the reference vocabulary present in the candidate.

    Asymmetric on purpose: a long abstract should not be penalised for containing words
    the plan never mentioned.
    """
    if not reference or not candidate:
        return 0.0
    return len(candidate & reference) / len(reference)
