"""Retraction-injection benchmark for conclusion maintenance.

Given a maintained topic, flip K of its supporting papers (seeded) to retracted, withdrawn,
corrected or expression-of-concern, run each strategy on its own copy of the topic, and
score the result against ground truth computed from the dependency graph *before*
anything propagates:

* a conclusion whose every supporting paper was retracted/withdrawn should be INVALIDATED;
* one that keeps some valid support should *not* be invalidated;
* one hit only by a correction or expression of concern should be NEEDS_REVIEW;
* one that depends on no flipped paper should not change at all.

Strategies share corpus, index, repositories and configuration; only the strategy
differs. ``racm`` is the system under test; ``b1`` to ``b4`` are the baselines.

Flipped statuses are *injected* observations: they say "the benchmark flipped this", not
"a registry said this", and nothing outside the evaluation harness can create them.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Awaitable, Callable, Iterable, Mapping
from datetime import UTC, datetime
from enum import StrEnum
from uuid import uuid4

from pydantic import BaseModel, Field

from researchagent.config.schemas import MaintenanceConfig, RetractionBenchmarkSettings
from researchagent.core.exceptions import ConfigurationError
from researchagent.core.interfaces.repositories import ConclusionRepository
from researchagent.core.logging import get_logger
from researchagent.models.integrity import IntegrityObservation, IntegrityStatus, ObservationBasis
from researchagent.models.maintenance import (
    ChangeType,
    ConclusionDiff,
    ConclusionRecord,
    MaintenanceMode,
    MaintenanceUsage,
)
from researchagent.schemas.maintenance import MaintenanceRequest
from researchagent.services.maintenance.dependency_graph import (
    DependencyGraphMapper,
    dependents_of,
)
from researchagent.services.maintenance.service import ConclusionMaintenanceService
from researchagent.workflows.maintenance import ConclusionGenerator, MaintenanceAgentFactory
from researchagent.workflows.maintenance_runner import MaintenanceRunner

logger = get_logger(__name__)

INJECTION_REGISTRY = "injection"


class Strategy(StrEnum):
    RACM = "racm"
    B1_FULL_REGENERATION = "b1_full_regeneration"
    B2_RETRIEVAL_FILTER = "b2_retrieval_filter"
    B3_LLM_JUDGE = "b3_llm_judge"
    B4_NO_PROPAGATION = "b4_no_propagation"


class Injection(BaseModel):
    model_config = {"frozen": True}

    paper_id: str = Field(min_length=1)
    status: IntegrityStatus


class InjectionPlan(BaseModel):
    model_config = {"frozen": True}

    seed: int
    k: int = Field(ge=1)
    statuses: tuple[IntegrityStatus, ...]
    injections: tuple[Injection, ...]

    def status_of(self, paper_id: str) -> IntegrityStatus | None:
        return next((item.status for item in self.injections if item.paper_id == paper_id), None)


class Dependence(StrEnum):
    # Every supporting paper was flipped to an excluding status.
    ONLY = "only"
    # Some, not all, supporting papers were flipped to an excluding status.
    PARTIAL = "partial"
    # Only correction / expression-of-concern flips reach it.
    REVIEW_ONLY = "review_only"
    # No flipped paper reaches it.
    NONE = "none"


class GroundTruth(BaseModel):
    model_config = {"frozen": True}

    conclusion_id: str
    dependence: Dependence
    flipped_papers: tuple[str, ...] = ()


class RetractionMetrics(BaseModel):
    """None means "not defined for this strategy or this sample", never zero."""

    model_config = {"frozen": True}

    conclusions: int = Field(ge=0)
    invalidation_precision: float | None = None
    invalidation_recall: float | None = None
    over_invalidation_rate: float | None = None
    wrongful_invalidations: int | None = None
    needs_review_correctness: float | None = None
    leakage_rate: float | None = None
    report_conclusions: int = 0
    leaking_conclusions: int = 0


class StrategyResult(BaseModel):
    model_config = {"frozen": True}

    strategy: Strategy
    metrics: RetractionMetrics
    usage: MaintenanceUsage
    diff_id: str | None = None
    notes: tuple[str, ...] = ()


class BenchmarkIdentity(BaseModel):
    """What must match for two reports to be comparable."""

    model_config = {"frozen": True}

    topic_id: str
    corpus_fingerprint: str
    config_digest: str
    seed: int
    k: int
    statuses: tuple[IntegrityStatus, ...]
    grounding_floor: float | None = None


class BenchmarkReport(BaseModel):
    model_config = {"frozen": True}

    label: str
    identity: BenchmarkIdentity
    plan: InjectionPlan
    truth: tuple[GroundTruth, ...]
    results: tuple[StrategyResult, ...]
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def result(self, strategy: Strategy) -> StrategyResult | None:
        return next((item for item in self.results if item.strategy is strategy), None)


# --- injection and ground truth -----------------------------------------------------------


def plan_injection(
    records: Iterable[ConclusionRecord],
    *,
    k: int,
    seed: int,
    statuses: tuple[IntegrityStatus, ...],
) -> InjectionPlan:
    """Choose K supporting papers and their flipped statuses, reproducibly."""
    papers = sorted({pid for record in records if record.is_live for pid in record.paper_ids})
    if k > len(papers):
        raise ConfigurationError(
            "K exceeds the number of supporting papers", k=k, papers=len(papers)
        )
    rng = random.Random(seed)  # noqa: S311 - reproducible sampling, not security
    chosen = sorted(rng.sample(papers, k))
    return InjectionPlan(
        seed=seed,
        k=k,
        statuses=statuses,
        injections=tuple(
            Injection(paper_id=paper_id, status=statuses[rng.randrange(len(statuses))])
            for paper_id in chosen
        ),
    )


def injected_observations(plan: InjectionPlan) -> tuple[IntegrityObservation, ...]:
    return tuple(
        IntegrityObservation(
            paper_id=item.paper_id,
            identifier=item.paper_id,
            status=item.status,
            basis=ObservationBasis.INJECTED,
            registry=INJECTION_REGISTRY,
            observation=(
                f"benchmark seed {plan.seed} flipped {item.paper_id} to {item.status.value}"
            ),
        )
        for item in plan.injections
    )


def ground_truth(
    records: Iterable[ConclusionRecord],
    plan: InjectionPlan,
    *,
    excluded_statuses: Iterable[IntegrityStatus],
    review_statuses: Iterable[IntegrityStatus],
    schema_version: str = "racm1",
) -> tuple[GroundTruth, ...]:
    """Expected dependence, from the dependency graph, before anything propagates."""
    live = tuple(record for record in records if record.is_live)
    graph = DependencyGraphMapper(schema_version=schema_version).build("ground-truth", live)
    reached = dependents_of(graph, [item.paper_id for item in plan.injections])
    excluding, reviewing = set(excluded_statuses), set(review_statuses)

    truth: list[GroundTruth] = []
    for record in sorted(live, key=lambda item: item.conclusion_id):
        flipped = reached.get(record.conclusion_id, frozenset())
        excluded = {pid for pid in flipped if plan.status_of(pid) in excluding}
        review = {pid for pid in flipped if plan.status_of(pid) in reviewing}
        if excluded and set(record.paper_ids) <= excluded:
            dependence = Dependence.ONLY
        elif excluded:
            dependence = Dependence.PARTIAL
        elif review:
            dependence = Dependence.REVIEW_ONLY
        else:
            dependence = Dependence.NONE
        truth.append(
            GroundTruth(
                conclusion_id=record.conclusion_id,
                dependence=dependence,
                flipped_papers=tuple(sorted(flipped)),
            )
        )
    return tuple(truth)


# --- scoring --------------------------------------------------------------------------------


def score(
    truth: Iterable[GroundTruth],
    changes: Mapping[str, ChangeType] | None,
    report: Iterable[tuple[str, ...]],
    excluded_papers: Iterable[str],
) -> RetractionMetrics:
    """Metrics from per-conclusion change types and the papers each reported conclusion
    rests on.

    ``changes`` is None for strategies that do not track individual conclusions (full
    regeneration): their invalidation metrics are undefined, not zero.
    """
    entries = tuple(truth)
    excluded = set(excluded_papers)
    reported = tuple(report)
    leaking = sum(1 for papers in reported if excluded & set(papers))
    leakage = round(leaking / len(reported), 6) if reported else None

    if changes is None:
        return RetractionMetrics(
            conclusions=len(entries),
            leakage_rate=leakage,
            report_conclusions=len(reported),
            leaking_conclusions=leaking,
        )

    predicted = {cid for cid, kind in changes.items() if kind is ChangeType.INVALIDATED}
    actual = {entry.conclusion_id for entry in entries if entry.dependence is Dependence.ONLY}
    true_positive = len(predicted & actual)
    unaffected = [entry.conclusion_id for entry in entries if entry.dependence is Dependence.NONE]
    review_only = [
        entry.conclusion_id for entry in entries if entry.dependence is Dependence.REVIEW_ONLY
    ]
    should_survive = {
        entry.conclusion_id
        for entry in entries
        if entry.dependence in (Dependence.PARTIAL, Dependence.REVIEW_ONLY, Dependence.NONE)
    }

    return RetractionMetrics(
        conclusions=len(entries),
        invalidation_precision=_ratio(true_positive, len(predicted)),
        invalidation_recall=_ratio(true_positive, len(actual)),
        over_invalidation_rate=_ratio(
            sum(1 for cid in unaffected if cid in changes), len(unaffected)
        ),
        wrongful_invalidations=len(predicted & should_survive),
        needs_review_correctness=_ratio(
            sum(1 for cid in review_only if changes.get(cid) is ChangeType.NEEDS_REVIEW),
            len(review_only),
        ),
        leakage_rate=leakage,
        report_conclusions=len(reported),
        leaking_conclusions=leaking,
    )


def _ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


# --- running ------------------------------------------------------------------------------

# (paper ids, run id) -> observations a model asserted. Baseline B3 only.
JudgeCallable = Callable[[tuple[str, ...], str], Awaitable[tuple[IntegrityObservation, ...]]]


class RetractionInjectionBenchmark:
    """Runs every requested strategy on its own clone of a maintained topic."""

    def __init__(
        self,
        service: ConclusionMaintenanceService,
        agent_for: MaintenanceAgentFactory,
        generate: ConclusionGenerator,
        settings: RetractionBenchmarkSettings,
        *,
        judge: JudgeCallable | None = None,
        config_digest: str = "",
    ) -> None:
        self._service = service
        self._agent_for = agent_for
        self._generate = generate
        self._settings = settings
        self._judge = judge
        self._config_digest = config_digest

    async def run(
        self,
        topic_id: str,
        *,
        label: str,
        strategies: Iterable[Strategy] | None = None,
        grounding_floor: float | None = None,
    ) -> BenchmarkReport:
        conclusions = self._service.conclusions
        records = await conclusions.for_topic(topic_id)
        settings = self._settings
        plan = plan_injection(records, k=settings.k, seed=settings.seed, statuses=settings.statuses)
        propagation = self._service.config.propagation
        truth = ground_truth(
            records,
            plan,
            excluded_statuses=propagation.excluded_statuses,
            review_statuses=propagation.review_statuses,
            schema_version=self._service.config.dependency_graph.schema_version,
        )
        excluded = tuple(
            item.paper_id
            for item in plan.injections
            if item.status in propagation.excluded_statuses
        )

        chosen = tuple(strategies or (Strategy(name) for name in settings.strategies))
        results: list[StrategyResult] = []
        for strategy in chosen:
            clone = await clone_topic(
                conclusions, topic_id, f"{topic_id}~{label}~{strategy.value}~{uuid4().hex[:6]}"
            )
            results.append(await self._run_one(strategy, clone, plan, truth, excluded))
            logger.info("benchmark_strategy_done", strategy=strategy.value, label=label)

        return BenchmarkReport(
            label=label,
            identity=BenchmarkIdentity(
                topic_id=topic_id,
                corpus_fingerprint=corpus_fingerprint(records),
                config_digest=self._config_digest,
                seed=settings.seed,
                k=settings.k,
                statuses=settings.statuses,
                grounding_floor=grounding_floor,
            ),
            plan=plan,
            truth=truth,
            results=tuple(results),
        )

    async def _run_one(
        self,
        strategy: Strategy,
        topic_id: str,
        plan: InjectionPlan,
        truth: tuple[GroundTruth, ...],
        excluded: tuple[str, ...],
    ) -> StrategyResult:
        injected = injected_observations(plan)
        clone_truth = _rekey(truth, topic_id)
        base = self._service.config

        if strategy in (Strategy.B1_FULL_REGENERATION, Strategy.B2_RETRIEVAL_FILTER):
            statuses = (
                base.propagation.excluded_statuses
                if strategy is Strategy.B2_RETRIEVAL_FILTER
                else ()
            )
            variant = base.model_copy(
                update={
                    "full_rerun": base.full_rerun.model_copy(update={"exclude_statuses": statuses})
                }
            )
            diff = await self._runner(variant).run(
                topic_id,
                MaintenanceRequest(mode=MaintenanceMode.FULL_RERUN),
                injected=injected,
                skip_registry=True,
            )
            regenerated = (
                await self._service.conclusions.for_topic(diff.regenerated_topic_id)
                if diff.regenerated_topic_id
                else ()
            )
            metrics = score(
                clone_truth, None, [r.paper_ids for r in regenerated if r.is_live], excluded
            )
            return StrategyResult(
                strategy=strategy, metrics=metrics, usage=diff.usage, diff_id=diff.id
            )

        notes: list[str] = []
        usage_extra = MaintenanceUsage()
        if strategy is Strategy.B3_LLM_JUDGE:
            if self._judge is None:
                raise ConfigurationError("Baseline B3 needs a judge; none was supplied")
            records = await self._service.conclusions.for_topic(topic_id)
            papers = tuple(sorted({pid for r in records if r.is_live for pid in r.paper_ids}))
            judge_run = f"{topic_id}~judge"
            injected = await self._judge(papers, judge_run)
            usage_extra = self._usage_of(judge_run)
            notes.append(
                f"model asserted a non-current status for {len(injected)} of {len(papers)} papers"
            )

        config: MaintenanceConfig = base
        if strategy is Strategy.B4_NO_PROPAGATION:
            config = base.model_copy(
                update={"propagation": base.propagation.model_copy(update={"enabled": False})}
            )
        config = config.model_copy(
            update={"new_papers": config.new_papers.model_copy(update={"detect": False})}
        )
        diff = await self._runner(config).run(
            topic_id,
            MaintenanceRequest(mode=MaintenanceMode.INCREMENTAL),
            injected=injected,
            skip_registry=True,
        )
        after = await self._service.conclusions.for_topic(topic_id)
        metrics = score(
            clone_truth,
            _changes(diff),
            [record.paper_ids for record in after if record.is_live],
            excluded,
        )
        return StrategyResult(
            strategy=strategy,
            metrics=metrics,
            usage=_add(diff.usage, usage_extra),
            diff_id=diff.id,
            notes=tuple(notes),
        )

    def _runner(self, config: MaintenanceConfig) -> MaintenanceRunner:
        return MaintenanceRunner(self._service.with_config(config), self._agent_for, self._generate)

    def _usage_of(self, run_id: str) -> MaintenanceUsage:
        tracker = self._service.usage
        if tracker is None:
            return MaintenanceUsage()
        summary = tracker.summary(run_id)
        return MaintenanceUsage(
            llm_calls=len(summary.calls),
            prompt_tokens=summary.prompt_tokens,
            completion_tokens=summary.completion_tokens,
            total_tokens=summary.total_tokens,
            estimated_cost_usd=summary.estimated_cost_usd if summary.calls else 0.0,
        )


async def clone_topic(conclusions: ConclusionRepository, topic_id: str, clone_id: str) -> str:
    """A private copy of a topic's latest conclusions, so strategies cannot interfere."""
    topic = await conclusions.get_topic(topic_id)
    if topic is None:
        raise ConfigurationError("No such topic to benchmark", topic_id=topic_id)
    await conclusions.save_topic(topic.model_copy(update={"topic_id": clone_id}))
    for record in await conclusions.for_topic(topic_id):
        await conclusions.append(
            record.model_copy(
                update={
                    "conclusion_id": _clone_id(record.conclusion_id, clone_id),
                    "topic_id": clone_id,
                    "version": 1,
                    "previous_version": None,
                }
            )
        )
    return clone_id


def corpus_fingerprint(records: Iterable[ConclusionRecord]) -> str:
    payload = "|".join(
        sorted(f"{r.conclusion_id}@{r.version}:{','.join(r.evidence_ids)}" for r in records)
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def config_digest(*configs: BaseModel) -> str:
    payload = json.dumps([config.model_dump(mode="json") for config in configs], sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def comparable(left: BenchmarkReport, right: BenchmarkReport) -> tuple[str, ...]:
    """Identity fields on which two reports disagree. Empty means comparable."""
    a, b = left.identity.model_dump(), right.identity.model_dump()
    return tuple(field for field in a if a[field] != b[field])


def _clone_id(conclusion_id: str, clone_id: str) -> str:
    return f"{conclusion_id}@{clone_id}"


def _rekey(truth: tuple[GroundTruth, ...], clone_id: str) -> tuple[GroundTruth, ...]:
    return tuple(
        entry.model_copy(update={"conclusion_id": _clone_id(entry.conclusion_id, clone_id)})
        for entry in truth
    )


def _changes(diff: ConclusionDiff) -> dict[str, ChangeType]:
    return {change.conclusion_id: change.change_type for change in diff.changes}


def _add(left: MaintenanceUsage, right: MaintenanceUsage) -> MaintenanceUsage:
    costs = (left.estimated_cost_usd, right.estimated_cost_usd)
    return MaintenanceUsage(
        llm_calls=left.llm_calls + right.llm_calls,
        prompt_tokens=left.prompt_tokens + right.prompt_tokens,
        completion_tokens=left.completion_tokens + right.completion_tokens,
        total_tokens=left.total_tokens + right.total_tokens,
        estimated_cost_usd=None if None in costs else round(sum(c or 0.0 for c in costs), 8),
        latency_ms=left.latency_ms + right.latency_ms,
    )
