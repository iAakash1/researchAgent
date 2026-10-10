"""Evaluation harness for conclusion maintenance. Offline: injected statuses, a scripted
registry, a fake LLM and a stub reasoning loop. Metric arithmetic is checked against
small cases computed by hand."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from researchagent.config.schemas import RetractionBenchmarkSettings
from researchagent.container import Container
from researchagent.core.exceptions import ConfigurationError
from researchagent.core.prompts import PromptLibrary
from researchagent.evaluation.gold import ReviewStatus
from researchagent.evaluation.llm_judge import LLMRetractionJudge, RetractionJudgementDraft
from researchagent.evaluation.real_retractions import (
    RealRetractionEntry,
    RealRetractionSet,
    check_against_registry,
)
from researchagent.evaluation.retraction import (
    BenchmarkReport,
    Dependence,
    GroundTruth,
    RetractionInjectionBenchmark,
    Strategy,
    comparable,
    ground_truth,
    plan_injection,
    score,
)
from researchagent.evaluation.sensitivity import sweep
from researchagent.evaluation.temporal import (
    ReplayWindow,
    TemporalAnnotation,
    TemporalGoldSet,
    score_replay,
    split_corpus,
)
from researchagent.models.integrity import IntegrityObservation, IntegrityStatus, ObservationBasis
from researchagent.models.knowledge import KnowledgeObject
from researchagent.models.maintenance import (
    ChangeTrigger,
    ChangeType,
    ConclusionChange,
    ConclusionDiff,
    MaintenanceMode,
    TriggerKind,
)
from researchagent.models.paper import Paper, SourceName
from researchagent.services.llm_service import BoundLLM
from tests.conftest import FakeLLMProvider, ScriptedIntegrityRegistry
from tests.maintenance.conftest import conclusion, paper, support
from tests.maintenance.test_workflow import (  # noqa: F401 - fixture
    TOPIC,
    answer,
    script_llm,
    seeded,
    stub_generator,
)

REPO = Path(__file__).resolve().parents[2]
P1, P2, P3, P4 = paper("p1"), paper("p2"), paper("p3"), paper("p4")
EXCLUDING = (IntegrityStatus.RETRACTED, IntegrityStatus.WITHDRAWN)
REVIEWING = (IntegrityStatus.EXPRESSION_OF_CONCERN, IntegrityStatus.CORRECTED)


# --- injection and ground truth -----------------------------------------------------------


def test_injection_is_seeded_and_bounded() -> None:
    records = [conclusion("C1", support(P1), support(P2)), conclusion("C2", support(P3))]
    statuses = (IntegrityStatus.RETRACTED, IntegrityStatus.CORRECTED)

    first = plan_injection(records, k=2, seed=7, statuses=statuses)
    again = plan_injection(records, k=2, seed=7, statuses=statuses)

    assert first == again
    assert len({item.paper_id for item in first.injections}) == 2
    assert all(item.status in statuses for item in first.injections)
    with pytest.raises(ConfigurationError):
        plan_injection(records, k=4, seed=7, statuses=statuses)


def test_ground_truth_from_the_dependency_graph() -> None:
    records = [
        conclusion("C-only", support(P1)),
        conclusion("C-partial", support(P1, 1), support(P2)),
        conclusion("C-review", support(P3)),
        conclusion("C-none", support(P4)),
    ]
    plan = plan_injection(records, k=1, seed=0, statuses=(IntegrityStatus.RETRACTED,))
    plan = plan.model_copy(
        update={
            "k": 2,
            "injections": (
                plan.injections[0].model_copy(update={"paper_id": P1}),
                plan.injections[0].model_copy(
                    update={"paper_id": P3, "status": IntegrityStatus.EXPRESSION_OF_CONCERN}
                ),
            ),
        }
    )

    truth = {
        item.conclusion_id: item.dependence
        for item in ground_truth(
            records, plan, excluded_statuses=EXCLUDING, review_statuses=REVIEWING
        )
    }

    assert truth == {
        "C-only": Dependence.ONLY,
        "C-partial": Dependence.PARTIAL,
        "C-review": Dependence.REVIEW_ONLY,
        "C-none": Dependence.NONE,
    }


# --- metric arithmetic (computed by hand) --------------------------------------------------

TRUTH = (
    GroundTruth(conclusion_id="C1", dependence=Dependence.ONLY),
    GroundTruth(conclusion_id="C2", dependence=Dependence.PARTIAL),
    GroundTruth(conclusion_id="C3", dependence=Dependence.REVIEW_ONLY),
    GroundTruth(conclusion_id="C4", dependence=Dependence.NONE),
)


def test_metric_arithmetic() -> None:
    changes = {
        "C1": ChangeType.INVALIDATED,  # true positive
        "C2": ChangeType.INVALIDATED,  # false positive, and a wrongful invalidation
        "C3": ChangeType.NEEDS_REVIEW,  # correct
        "C4": ChangeType.KEPT,  # an unaffected conclusion changed: over-invalidation
    }
    report = [(P2,), (P3,), (P1, P2)]  # one of three still rests on retracted P1

    metrics = score(TRUTH, changes, report, [P1])

    assert metrics.invalidation_precision == 0.5  # 1 / 2 predicted
    assert metrics.invalidation_recall == 1.0  # 1 / 1 actual
    assert metrics.over_invalidation_rate == 1.0  # 1 / 1 unaffected
    assert metrics.wrongful_invalidations == 1
    assert metrics.needs_review_correctness == 1.0  # 1 / 1 review-only
    assert metrics.leakage_rate == 0.333333  # 1 / 3 reported
    assert (metrics.report_conclusions, metrics.leaking_conclusions) == (3, 1)


def test_undefined_metrics_are_none_not_zero() -> None:
    nothing_predicted = score(TRUTH, {}, [], [P1])
    assert nothing_predicted.invalidation_precision is None
    assert nothing_predicted.invalidation_recall == 0.0
    assert nothing_predicted.leakage_rate is None  # empty report

    untracked = score(TRUTH, None, [(P1,)], [P1])
    assert untracked.invalidation_precision is None and untracked.invalidation_recall is None
    assert untracked.over_invalidation_rate is None and untracked.wrongful_invalidations is None
    assert untracked.leakage_rate == 1.0


# --- end to end over every strategy --------------------------------------------------------


async def test_benchmark_runs_every_strategy_on_its_own_copy(
    container: Container,
    seeded: dict[str, KnowledgeObject],  # noqa: F811
    fake_provider: FakeLLMProvider,
) -> None:
    script_llm(fake_provider, seeded)  # RACM's one re-verification and one review
    records = await container.maintenance.conclusions.for_topic(TOPIC)
    statuses = (IntegrityStatus.RETRACTED,)
    seed = next(
        s
        for s in range(100)
        if plan_injection(records, k=1, seed=s, statuses=statuses).injections[0].paper_id == P1
    )
    judged: list[tuple[str, ...]] = []

    async def judge(papers: tuple[str, ...], run_id: str) -> tuple[IntegrityObservation, ...]:
        judged.append(papers)
        return ()  # the model believes nothing is retracted

    maintenance = container.maintenance
    benchmark = RetractionInjectionBenchmark(
        maintenance.service,
        maintenance.agent_for,
        stub_generator(container, seeded, []),
        RetractionBenchmarkSettings(k=1, seed=seed, statuses=statuses),
        judge=judge,
        config_digest="test",
    )

    report = await benchmark.run(TOPIC, label="t")

    truth = {item.conclusion_id: item.dependence for item in report.truth}
    assert truth == {
        "C-mixed": Dependence.PARTIAL,
        "C-other": Dependence.NONE,
        "C-solo": Dependence.ONLY,
    }

    racm = report.result(Strategy.RACM)
    assert racm is not None
    assert racm.metrics.invalidation_precision == 1.0
    assert racm.metrics.invalidation_recall == 1.0
    assert racm.metrics.over_invalidation_rate == 0.0
    assert racm.metrics.leakage_rate == 0.0
    assert racm.usage.total_tokens == 240

    b4 = report.result(Strategy.B4_NO_PROPAGATION)
    assert b4 is not None
    assert b4.metrics.invalidation_recall == 0.0
    assert b4.metrics.leakage_rate == 0.666667  # C-solo and C-mixed still cite P1

    b1 = report.result(Strategy.B1_FULL_REGENERATION)
    b2 = report.result(Strategy.B2_RETRIEVAL_FILTER)
    assert b1 is not None and b2 is not None
    assert b1.metrics.invalidation_precision is None  # it does not track conclusions
    assert b1.metrics.leakage_rate == 1.0  # regenerated over a corpus that still has P1
    assert b2.metrics.leakage_rate == 0.0  # P1 dropped at retrieval time
    assert b1.usage.total_tokens == 1200 and b1.usage.estimated_cost_usd == pytest.approx(0.01)

    b3 = report.result(Strategy.B3_LLM_JUDGE)
    assert b3 is not None
    assert judged and P1 in judged[0]
    assert b3.metrics.invalidation_recall == 0.0
    assert b3.metrics.leakage_rate == 0.666667

    # The original topic is untouched: every strategy ran on its own copy.
    for conclusion_id in ("C-solo", "C-mixed", "C-other"):
        assert len(await maintenance.conclusions.history(conclusion_id)) == 1


def test_reports_with_different_setups_are_not_comparable() -> None:
    from researchagent.evaluation.retraction import BenchmarkIdentity, InjectionPlan

    def report(seed: int) -> BenchmarkReport:
        return BenchmarkReport(
            label=f"s{seed}",
            identity=BenchmarkIdentity(
                topic_id="T",
                corpus_fingerprint="abc",
                config_digest="def",
                seed=seed,
                k=1,
                statuses=(IntegrityStatus.RETRACTED,),
            ),
            plan=InjectionPlan(
                seed=seed, k=1, statuses=(IntegrityStatus.RETRACTED,), injections=()
            ),
            truth=(),
            results=(),
        )

    assert comparable(report(1), report(1)) == ()
    assert comparable(report(1), report(2)) == ("seed",)


# --- real retractions: schema and registry check -------------------------------------------


def test_shipped_template_is_empty_and_valid() -> None:
    template = RealRetractionSet.load(REPO / "evaluation/gold/real_retractions.template.json")
    assert template.entries == ()
    with pytest.raises(ValidationError):
        RealRetractionEntry(doi="not-a-doi", expected_status=IntegrityStatus.RETRACTED, source="x")


async def test_registry_agreement_counts_failures_separately(container: Container) -> None:
    registry = container.maintenance.registry
    assert isinstance(registry, ScriptedIntegrityRegistry)
    registry.answers = {
        "10.5555/agree": answer("10.5555/agree", "retraction"),
        "10.5555/disagree": answer("10.5555/disagree"),
    }
    dataset = RealRetractionSet(
        version="t",
        entries=tuple(
            RealRetractionEntry(doi=doi, expected_status=IntegrityStatus.RETRACTED, source="test")
            for doi in ("10.5555/agree", "10.5555/disagree", "10.5555/unreachable")
        ),
    )

    report = await check_against_registry(dataset, container.maintenance.integrity)

    assert (report.agreed, report.disagreed, report.lookup_failures) == (1, 1, 1)
    assert report.agreement_rate == 0.5


# --- temporal replay ----------------------------------------------------------------------


def test_corpus_split_by_year() -> None:
    papers = [
        Paper(id=pid, title=pid, year=year, provider=SourceName.MANUAL)
        for pid, year in (
            ("old", 2021),
            ("edge", 2023),
            ("new", 2024),
            ("late", 2026),
            ("none", None),
        )
    ]
    split = split_corpus(papers, ReplayWindow(frozen_through_year=2023, added_through_year=2025))

    assert split.frozen == ("edge", "old") and split.added == ("new",)
    assert split.after_window == ("late",) and split.undated == ("none",)
    with pytest.raises(ValidationError):
        ReplayWindow(frozen_through_year=2025, added_through_year=2025)


def test_replay_scores_only_reviewed_annotations() -> None:
    template = TemporalGoldSet.load(REPO / "evaluation/gold/temporal_replay.template.json")
    assert template.annotations == ()

    def annotate(cid: str, expected: ChangeType | None, status: ReviewStatus) -> TemporalAnnotation:
        return TemporalAnnotation(
            conclusion_id=cid, expected_change=expected, rationale="r", annotator="a", status=status
        )

    gold = template.model_copy(
        update={
            "annotations": (
                annotate("C1", ChangeType.NEWLY_CONTRADICTED, ReviewStatus.REVIEWED),
                annotate("C2", None, ReviewStatus.REVIEWED),
                annotate("C3", ChangeType.KEPT, ReviewStatus.REVIEWED),
                annotate("C4", ChangeType.KEPT, ReviewStatus.DRAFT),
            )
        }
    )
    trigger = ChangeTrigger(kind=TriggerKind.NEW_PAPER, detail="new paper")
    diff = ConclusionDiff(
        topic_id="T",
        maintenance_run_id="M",
        mode=MaintenanceMode.INCREMENTAL,
        changes=(
            ConclusionChange(
                conclusion_id="C1",
                before_version=1,
                after_version=2,
                change_type=ChangeType.NEWLY_CONTRADICTED,
                triggers=(trigger,),
            ),
            ConclusionChange(
                conclusion_id="C3",
                before_version=1,
                after_version=2,
                change_type=ChangeType.WEAKENED,
                triggers=(trigger,),
            ),
        ),
    )

    result = score_replay(gold, diff)

    assert (result.reviewed, result.drafts, result.matched) == (3, 1, 2)  # C1 and C2 match
    assert result.accuracy == 0.666667
    assert result.confusion["kept"] == {"weakened": 1}
    assert result.confusion["unchanged"] == {"unchanged": 1}


# --- sensitivity sweep --------------------------------------------------------------------


async def test_sensitivity_sweep_runs_each_floor() -> None:
    from researchagent.evaluation.retraction import (
        BenchmarkIdentity,
        InjectionPlan,
        RetractionMetrics,
        StrategyResult,
    )
    from researchagent.models.maintenance import MaintenanceUsage

    prepared: list[float] = []

    async def prepare(floor: float) -> str:
        prepared.append(floor)
        return f"T-{floor}"

    async def run(topic_id: str, floor: float) -> BenchmarkReport:
        return BenchmarkReport(
            label=topic_id,
            identity=BenchmarkIdentity(
                topic_id=topic_id,
                corpus_fingerprint="x",
                config_digest="y",
                seed=1,
                k=1,
                statuses=(IntegrityStatus.RETRACTED,),
                grounding_floor=floor,
            ),
            plan=InjectionPlan(seed=1, k=1, statuses=(IntegrityStatus.RETRACTED,), injections=()),
            truth=(),
            results=(
                StrategyResult(
                    strategy=Strategy.RACM,
                    metrics=RetractionMetrics(conclusions=1, leakage_rate=floor / 10),
                    usage=MaintenanceUsage(total_tokens=int(floor * 100)),
                ),
            ),
        )

    rows = await sweep((0.75, 0.95), prepare, run)

    assert prepared == [0.75, 0.95]
    assert [(row.floor, row.topic_id, row.total_tokens) for row in rows] == [
        (0.75, "T-0.75", 75),
        (0.95, "T-0.95", 95),
    ]


# --- baseline B3 judge --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("answer_status", "expected"),
    [("retracted", IntegrityStatus.RETRACTED), ("current", None), ("no idea", None)],
)
async def test_llm_judge_assertions_are_tagged_as_injected(
    model_catalog: object, answer_status: str, expected: IntegrityStatus | None
) -> None:
    provider = FakeLLMProvider(
        structured=RetractionJudgementDraft(status=answer_status, reasoning="r")
    )
    llm = BoundLLM("fast", model_catalog.spec_for("fast"), provider)  # type: ignore[attr-defined]
    judge = LLMRetractionJudge(llm, PromptLibrary(REPO / "prompts").load("retraction_judge", "v1"))

    found = await judge((P1,), "run")

    if expected is None:
        assert found == ()
    else:
        (item,) = found
        assert item.status is expected
        assert item.basis is ObservationBasis.INJECTED
        assert item.registry == "llm_judge:fast"
    assert "10.5555/p1" in provider.calls[0][-1].content
