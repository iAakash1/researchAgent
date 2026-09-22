from __future__ import annotations

from datetime import UTC, datetime

from researchagent.config.schemas import RankingConfig, RankingWeights
from researchagent.models.paper import Paper, SourceName
from researchagent.models.research import (
    QuestionPriority,
    ResearchPlan,
    ResearchQuestion,
    SearchStrategy,
)
from researchagent.services.ranking import HeuristicScorer, RelevanceDecision

THIS_YEAR = datetime.now(UTC).year


def a_plan() -> ResearchPlan:
    return ResearchPlan(
        topic="Metastable failures in distributed systems",
        framing="A review of metastable failure modes and their triggers in large systems.",
        research_questions=[
            ResearchQuestion(
                id="RQ1",
                question="What triggers metastable failures in distributed systems?",
                rationale="Triggers determine which mitigations are even applicable.",
                priority=QuestionPriority.HIGH,
                keywords=["metastable", "overload"],
            )
        ],
        strategy=SearchStrategy(queries=["metastable failure distributed systems"]),
    )


def fake_review_plan() -> ResearchPlan:
    return ResearchPlan(
        topic="Detecting deceptive online consumer reviews",
        framing="A review of methods for identifying deceptive opinions and review manipulation.",
        research_questions=[
            ResearchQuestion(
                id="RQ1",
                question="Which methods detect deceptive online consumer reviews?",
                rationale="The corpus must focus on opinion spam rather than generic detection.",
                priority=QuestionPriority.HIGH,
                keywords=["deceptive reviews", "opinion spam", "review manipulation"],
            )
        ],
        strategy=SearchStrategy(
            queries=["fake online reviews", "deceptive opinion spam detection"]
        ),
    )


def paper(title: str, **overrides: object) -> Paper:
    return Paper.model_validate(
        {"id": f"x:{title[:10]}", "title": title, "provider": SourceName.ARXIV} | overrides
    )


def test_on_topic_paper_outranks_off_topic() -> None:
    scorer = HeuristicScorer()
    plan = a_plan()

    relevant = scorer.score(paper("Metastable failures in distributed systems"), plan)
    irrelevant = scorer.score(paper("A survey of medieval pottery glazing"), plan)

    assert relevant.score > irrelevant.score


def test_signals_sum_to_the_score() -> None:
    scored = HeuristicScorer().score(
        paper("Metastable failures", year=2024, citation_count=100), a_plan()
    )

    assert scored.score == round(sum(scored.signals.values()), 6)
    assert set(scored.signals) == {
        "title_match",
        "abstract_match",
        "keyword_overlap",
        "recency",
        "citations",
    }


def test_abstract_contributes_when_the_title_does_not() -> None:
    scorer = HeuristicScorer()
    plan = a_plan()

    bare = scorer.score(paper("An empirical study"), plan)
    with_abstract = scorer.score(
        paper(
            "An empirical study",
            abstract="We study metastable failure triggers in large distributed systems.",
        ),
        plan,
    )

    assert with_abstract.score > bare.score
    assert with_abstract.signals["abstract_match"] > 0


def test_recent_papers_outrank_old_ones() -> None:
    scorer = HeuristicScorer()
    plan = a_plan()
    title = "Metastable failures in distributed systems"

    recent = scorer.score(paper(title, year=THIS_YEAR), plan)
    old = scorer.score(paper(title, year=THIS_YEAR - 20), plan)

    assert recent.signals["recency"] > old.signals["recency"]


def test_missing_year_and_citations_score_neutrally_not_zero() -> None:
    """Manual papers and arXiv preprints report neither; they must not be buried."""
    scored = HeuristicScorer().score(paper("Metastable failures"), a_plan())

    assert scored.signals["recency"] > 0
    assert scored.signals["citations"] > 0


def test_zero_citations_is_distinguished_from_unknown() -> None:
    scorer = HeuristicScorer()
    plan = a_plan()
    title = "Metastable failures"

    unknown = scorer.score(paper(title), plan)
    explicit_zero = scorer.score(paper(title, citation_count=0), plan)

    assert unknown.signals["citations"] > explicit_zero.signals["citations"] == 0


def test_citation_influence_saturates() -> None:
    scorer = HeuristicScorer(RankingConfig(citation_saturation=500))
    plan = a_plan()
    title = "Metastable failures"

    big = scorer.score(paper(title, citation_count=500), plan)
    huge = scorer.score(paper(title, citation_count=50_000), plan)

    assert huge.signals["citations"] == big.signals["citations"]


def test_weights_change_the_outcome() -> None:
    plan = a_plan()
    recent_relevant = paper(
        "Metastable failures in distributed systems recent study", year=THIS_YEAR
    )
    old_relevant = paper(
        "Metastable failures in distributed systems legacy study", year=THIS_YEAR - 15
    )

    recency_only = HeuristicScorer(
        RankingConfig(
            weights=RankingWeights(
                title_match=0, abstract_match=0, keyword_overlap=0, recency=1, citations=0
            )
        )
    )
    ranked = recency_only.rank([old_relevant, recent_relevant], plan)

    assert ranked[0].paper.title == "Metastable failures in distributed systems recent study"


def test_rank_orders_and_limits() -> None:
    plan = a_plan()
    papers = [
        paper("Unrelated pottery study"),
        paper("Metastable failures in distributed systems", year=THIS_YEAR),
        paper("Distributed systems overload"),
    ]

    ranked = HeuristicScorer().rank(papers, plan, limit=2)

    assert len(ranked) == 2
    assert ranked[0].score >= ranked[1].score
    assert ranked[0].paper.title == "Metastable failures in distributed systems"


def test_ranking_is_deterministic() -> None:
    plan = a_plan()
    papers = [paper(f"Metastable failures variant {i}", year=2024) for i in range(5)]
    scorer = HeuristicScorer()

    first = [item.paper.title for item in scorer.rank(papers, plan)]
    second = [item.paper.title for item in scorer.rank(list(reversed(papers)), plan)]

    assert first == second


def test_score_stays_within_bounds() -> None:
    scored = HeuristicScorer().score(
        paper(
            "Metastable failures in distributed systems overload triggers",
            abstract="metastable overload distributed systems failures triggers",
            keywords=["metastable", "overload"],
            year=THIS_YEAR,
            citation_count=100_000,
        ),
        a_plan(),
    )

    assert 0.0 <= scored.score <= 1.0


def test_fake_review_relevance_rejects_unrelated_detection_papers() -> None:
    scorer = HeuristicScorer()
    plan = fake_review_plan()
    goal = "Fake online review detection"

    relevant = scorer.score(
        paper(
            "Fake online review detection using textual and behavioral signals",
            abstract="We identify deceptive consumer reviews on commerce platforms.",
        ),
        plan,
        research_goal=goal,
    )
    alzheimers = scorer.score(
        paper("Alzheimer's disease detection using deep learning"),
        plan,
        research_goal=goal,
    )
    medical_images = scorer.score(
        paper("Medical image segmentation with diffusion models"),
        plan,
        research_goal=goal,
    )
    personality = scorer.score(
        paper("Personality traits detection using neural networks"),
        plan,
        research_goal=goal,
    )

    assert relevant.relevance_decision is RelevanceDecision.DIRECT
    assert relevant.relevance_score >= 0.6
    assert alzheimers.relevance_decision is RelevanceDecision.IRRELEVANT
    assert medical_images.relevance_decision is RelevanceDecision.IRRELEVANT
    assert personality.relevance_decision is RelevanceDecision.IRRELEVANT
    assert relevant.relevance_score > alzheimers.relevance_score
    assert relevant.relevance_score > medical_images.relevance_score


def test_generic_method_overlap_is_not_topic_relevance() -> None:
    scored = HeuristicScorer().score(
        paper(
            "Detection and classification using deep machine learning",
            keywords=["detection", "machine learning", "deep learning"],
        ),
        fake_review_plan(),
        research_goal="Fake online review detection",
    )

    assert scored.relevance_score == 0
    assert scored.relevance_decision is RelevanceDecision.IRRELEVANT


def test_plan_concepts_allow_semantically_relevant_alternate_terminology() -> None:
    scored = HeuristicScorer().score(
        paper(
            "Opinion spam classification with behavioral signals",
            abstract="We identify deceptive user-generated product feedback.",
        ),
        fake_review_plan(),
        research_goal="Fake online review detection",
    )

    assert scored.relevance_decision is RelevanceDecision.DIRECT
    assert scored.relevance_signals["concept_alignment"] == 1


def test_broad_fake_news_detection_is_related_but_not_direct() -> None:
    scored = HeuristicScorer().score(
        paper("Fake news detection using transformer models"),
        fake_review_plan(),
        research_goal="Fake online review detection",
    )

    assert scored.relevance_decision is RelevanceDecision.RELATED
