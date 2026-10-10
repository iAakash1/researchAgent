"""Conclusion maintenance: the domain steps a maintenance run is made of.

The LangGraph workflow (``workflows/maintenance.py``) decides the order and calls agents;
everything it needs to *know* — which papers changed, which conclusions depend on them,
what each rule says, what gets written — lives here.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import UTC, datetime

from pydantic import BaseModel, Field

from researchagent.config.schemas import MaintenanceConfig
from researchagent.core.events import (
    ConclusionChangePayload,
    EventBus,
    EventType,
    MaintenancePayload,
)
from researchagent.core.exceptions import (
    ConclusionNotFoundError,
    GraphStoreError,
    RepositoryError,
    TopicNotFoundError,
)
from researchagent.core.interfaces.repositories import (
    ConclusionRepository,
    KnowledgeRepository,
    MaintenanceDiffRepository,
)
from researchagent.core.logging import get_logger
from researchagent.models.integrity import IntegrityChange, IntegrityObservation, IntegrityStatus
from researchagent.models.knowledge import KnowledgeObject
from researchagent.models.maintenance import (
    ChangeTrigger,
    ChangeType,
    ConclusionChange,
    ConclusionDiff,
    ConclusionRecord,
    MaintainedTopic,
    MaintenanceFailure,
    MaintenanceMode,
    MaintenanceUsage,
    TriggerKind,
)
from researchagent.models.reasoning import (
    FindingStatus,
    ResearchFinding,
    ReviewResult,
    VerificationResult,
)
from researchagent.schemas.maintenance import (
    ConclusionOutcome,
    MaintenanceState,
    PropagationInput,
    PropagationPlan,
    ReverificationRequest,
)
from researchagent.schemas.reasoning import ReasoningSession
from researchagent.services.integrity.service import IntegrityService
from researchagent.services.maintenance.dependency_graph import (
    DependencyGraphMapper,
    DependencyGraphStore,
)
from researchagent.services.maintenance.propagation import PropagationEngine
from researchagent.services.maintenance.recorder import ConclusionRecorder
from researchagent.services.maintenance.reverification import (
    ReverificationContext,
    ReverificationContextBuilder,
)
from researchagent.services.model_usage import ModelUsageTracker
from researchagent.utils.text import normalise

logger = get_logger(__name__)


class RefreshOutcome(BaseModel):
    model_config = {"frozen": True}

    changes: tuple[IntegrityChange, ...] = ()
    effective: dict[str, IntegrityObservation] = Field(default_factory=dict)
    lookup_failures: tuple[str, ...] = ()
    papers_checked: int = 0


class ConclusionMaintenanceService:
    name = "conclusion_maintenance"

    def __init__(
        self,
        conclusions: ConclusionRepository,
        diffs: MaintenanceDiffRepository,
        integrity: IntegrityService,
        recorder: ConclusionRecorder,
        engine: PropagationEngine,
        mapper: DependencyGraphMapper,
        graph_store: DependencyGraphStore,
        knowledge: KnowledgeRepository,
        contexts: ReverificationContextBuilder,
        config: MaintenanceConfig,
        *,
        usage: ModelUsageTracker | None = None,
        event_bus: EventBus | None = None,
    ) -> None:
        self._conclusions = conclusions
        self._diffs = diffs
        self._integrity = integrity
        self._recorder = recorder
        self._engine = engine
        self._mapper = mapper
        self._graph_store = graph_store
        self._knowledge = knowledge
        self._contexts = contexts
        self._config = config
        self._usage = usage
        self._event_bus = event_bus

    @property
    def config(self) -> MaintenanceConfig:
        return self._config

    def with_config(self, config: MaintenanceConfig) -> ConclusionMaintenanceService:
        """The same collaborators under another configuration — how the evaluation harness
        runs a baseline (propagation off, a full rerun with or without filtering) on the
        same corpus, index and repositories as the system under test."""
        return ConclusionMaintenanceService(
            self._conclusions,
            self._diffs,
            self._integrity,
            self._recorder,
            self._engine.with_settings(config.propagation),
            self._mapper,
            self._graph_store,
            self._knowledge,
            self._contexts,
            config,
            usage=self._usage,
            event_bus=self._event_bus,
        )

    @property
    def usage(self) -> ModelUsageTracker | None:
        return self._usage

    @property
    def conclusions(self) -> ConclusionRepository:
        return self._conclusions

    # --- reads --------------------------------------------------------------------------

    async def topic(self, topic_id: str) -> MaintainedTopic:
        topic = await self._conclusions.get_topic(topic_id)
        if topic is None:
            raise TopicNotFoundError("No saved topic with this id", topic_id=topic_id)
        return topic

    async def topics(self) -> tuple[MaintainedTopic, ...]:
        return await self._conclusions.list_topics()

    async def records(self, topic_id: str) -> tuple[ConclusionRecord, ...]:
        return await self._conclusions.for_topic(topic_id)

    async def history(self, conclusion_id: str) -> tuple[ConclusionRecord, ...]:
        history = await self._conclusions.history(conclusion_id)
        if not history:
            raise ConclusionNotFoundError("No conclusion with this id", conclusion_id=conclusion_id)
        return history

    async def latest_diff(self, topic_id: str) -> ConclusionDiff | None:
        await self.topic(topic_id)
        return await self._diffs.latest(topic_id)

    # --- refresh and detection ----------------------------------------------------------

    async def new_papers(
        self, topic: MaintainedTopic, requested: Iterable[str] = ()
    ) -> tuple[str, ...]:
        known = set(topic.known_paper_ids)
        wanted = tuple(requested)
        if wanted:
            return tuple(paper_id for paper_id in dict.fromkeys(wanted) if paper_id not in known)
        if not self._config.new_papers.detect:
            return ()
        trusted = await self._recorder.trusted_paper_ids()
        return tuple(paper_id for paper_id in trusted if paper_id not in known)

    async def refresh(
        self,
        topic: MaintainedTopic,
        records: Iterable[ConclusionRecord],
        new_paper_ids: Iterable[str],
        *,
        run_id: str,
        now: datetime | None = None,
        force: bool = False,
        injected: Iterable[IntegrityObservation] = (),
        skip_registry: bool = False,
    ) -> RefreshOutcome:
        """Integrity of every paper the topic's live conclusions (and new papers) rest on,
        compared against what this topic last absorbed."""
        papers = tuple(
            dict.fromkeys(
                [pid for record in records if record.is_live for pid in record.paper_ids]
                + list(new_paper_ids)
            )
        )
        supplied = {item.paper_id: item for item in injected}
        to_check = () if skip_registry else tuple(pid for pid in papers if pid not in supplied)
        refreshed = await self._integrity.refresh(to_check, now=now, force=force, run_id=run_id)

        effective: dict[str, IntegrityObservation] = {
            pid: observation
            for pid, observation in topic.integrity_baseline.items()
            if pid in papers
        }
        changes: list[IntegrityChange] = []
        failures: list[str] = []
        for item in refreshed:
            if item.lookup_failed:
                failures.append(f"{item.paper_id}: {item.observed.observation}")
            if item.effective is not None:
                effective[item.paper_id] = item.effective
            changes.append(
                IntegrityChange(
                    paper_id=item.paper_id,
                    before=topic.integrity_baseline.get(item.paper_id),
                    after=item.effective or item.observed,
                    cached=item.cached,
                )
            )
        for paper_id, observation in supplied.items():
            effective[paper_id] = observation
            changes.append(
                IntegrityChange(
                    paper_id=paper_id,
                    before=topic.integrity_baseline.get(paper_id),
                    after=observation,
                )
            )
        return RefreshOutcome(
            changes=tuple(changes),
            effective=effective,
            lookup_failures=tuple(failures),
            papers_checked=len(papers),
        )

    def excluded_papers(
        self, effective: Mapping[str, IntegrityObservation], statuses: Iterable[IntegrityStatus]
    ) -> frozenset[str]:
        wanted = set(statuses)
        return frozenset(pid for pid, item in effective.items() if item.status in wanted)

    # --- propagation --------------------------------------------------------------------

    async def plan(self, state: MaintenanceState) -> PropagationPlan:
        topic = _require(state.topic)
        excluded = self.excluded_papers(state.effective, self._config.propagation.excluded_statuses)
        live = tuple(record for record in state.records if record.is_live)

        objects: dict[str, KnowledgeObject] = {}
        for record in live:
            for link in record.supports:
                if link.knowledge_object_id not in objects:
                    found = await self._object(link.paper_id, link.knowledge_object_id)
                    if found is not None:
                        objects[found.id] = found
        new_objects: list[KnowledgeObject] = []
        for paper_id in state.new_paper_ids:
            if paper_id in excluded:
                continue  # a retracted newcomer neither supports nor contradicts anything
            stored = await self._knowledge.get(paper_id)
            if stored is not None and stored.is_trusted:
                new_objects.extend(stored.value.objects)

        papers = {link.paper_id for record in live for link in record.supports} | set(
            state.new_paper_ids
        )
        data = PropagationInput(
            records=state.records,
            graph=self._mapper.build(topic.topic_id, state.records, state.effective),
            changes=state.changes,
            effective=state.effective,
            new_objects=tuple(new_objects),
            supporting_objects=objects,
            dois={pid: await self._recorder.doi_for(pid) for pid in sorted(papers)},
        )
        return self._engine.plan(data)

    async def reverification_context(
        self, request: ReverificationRequest, topic: MaintainedTopic, *, run_id: str
    ) -> ReverificationContext:
        return await self._contexts.build(request, topic, run_id=run_id)

    def resolve(
        self,
        request: ReverificationRequest,
        verification: VerificationResult | None,
        *,
        run_id: str,
        failure: str | None = None,
    ) -> ConclusionOutcome:
        return self._engine.resolve(request, verification, run_id=run_id, failure=failure)

    def after_review(
        self,
        outcomes: Iterable[ConclusionOutcome],
        review: ReviewResult,
        finding_ids: Mapping[str, str],
    ) -> tuple[ConclusionOutcome, ...]:
        """A KEPT conclusion the reviewer rejects is flagged for review, never silently kept.

        ``finding_ids`` maps conclusion id to the re-verification finding the reviewer saw.
        """
        rejected = set(review.rejected_findings)
        updated: list[ConclusionOutcome] = []
        for outcome in outcomes:
            finding_id = finding_ids.get(outcome.record.conclusion_id)
            if outcome.change_type is not ChangeType.KEPT or finding_id not in rejected:
                updated.append(outcome)
                continue
            codes = sorted(
                {
                    issue.code
                    for issue in review.issues
                    if issue.finding_id == finding_id and issue.blocking
                }
            )
            updated.append(
                outcome.model_copy(
                    update={
                        "change_type": ChangeType.NEEDS_REVIEW,
                        "triggers": (
                            *outcome.triggers,
                            ChangeTrigger(
                                kind=TriggerKind.REVIEW,
                                detail=f"reviewer rejected the re-verified conclusion: {codes}",
                            ),
                        ),
                        "reasons": (*outcome.reasons, f"reviewer issues {codes}"),
                    }
                )
            )
        return tuple(updated)

    def accepted_findings(
        self,
        session: ReasoningSession | None,
        records: Iterable[ConclusionRecord],
    ) -> tuple[ResearchFinding, ...]:
        """Reviewer-accepted findings of a regeneration that do not restate a live conclusion."""
        if session is None:
            return ()
        existing = {normalise(record.statement) for record in records if record.is_live}
        return tuple(
            finding
            for finding in session.findings
            if finding.status is FindingStatus.VERIFIED
            and normalise(finding.statement) not in existing
        )

    # --- writing ------------------------------------------------------------------------

    async def finalize(self, state: MaintenanceState) -> ConclusionDiff:
        """Append every decided version, record new conclusions, advance the topic."""
        topic = _require(state.topic)
        run_id = state.run_id
        failures = list(state.failures)
        changes: list[ConclusionChange] = []
        held_back: set[str] = set()

        for outcome in state.outcomes:
            record, change = self._engine.apply(outcome, run_id=run_id)
            try:
                await self._conclusions.append(record)
            except RepositoryError as exc:
                failures.append(
                    MaintenanceFailure(
                        stage="commit",
                        code=exc.code,
                        message=exc.message,
                        conclusion_id=record.conclusion_id,
                    )
                )
                held_back.update(t.paper_id for t in outcome.triggers if t.paper_id)
                continue
            changes.append(change)
            await self._emit_change(topic.topic_id, change)

        regenerated_topic: MaintainedTopic | None = None
        if state.mode is MaintenanceMode.FULL_RERUN:
            regenerated_topic = MaintainedTopic(
                topic_id=f"{topic.topic_id}~rerun~{run_id}",
                goal=topic.goal,
                plan=topic.plan,
                known_paper_ids=await self._recorder.trusted_paper_ids(),
                integrity_baseline=dict(state.effective),
            )
            await self._conclusions.save_topic(regenerated_topic)
        target = regenerated_topic or topic
        new_changes, new_failures = await self._record_new(state, target)
        changes.extend(new_changes)
        failures.extend(new_failures)

        baseline = dict(topic.integrity_baseline)
        baseline.update(
            {
                pid: item
                for pid, item in state.effective.items()
                if item.is_informative and pid not in held_back
            }
        )
        known = tuple(dict.fromkeys((*topic.known_paper_ids, *state.new_paper_ids)))
        latest = await self._conclusions.for_topic(topic.topic_id)
        graph_version, graph_failure = await self._write_graph(topic.topic_id, latest, baseline)
        if graph_failure is not None:
            failures.append(graph_failure)
        await self._conclusions.save_topic(
            topic.model_copy(
                update={
                    "integrity_baseline": baseline,
                    "known_paper_ids": known,
                    "dependency_graph_version": graph_version or topic.dependency_graph_version,
                    "updated_at": datetime.now(UTC),
                }
            )
        )

        completed = datetime.now(UTC)
        plan = state.plan
        live_before = tuple(record for record in state.records if record.is_live)
        diff = ConclusionDiff(
            topic_id=topic.topic_id,
            maintenance_run_id=run_id,
            mode=state.mode,
            changes=tuple(changes),
            total_conclusions=len(state.records),
            affected_conclusions=(
                len(live_before)
                if state.mode is MaintenanceMode.FULL_RERUN
                else len(plan.affected_ids)
                if plan
                else 0
            ),
            papers_checked=state.papers_checked,
            integrity_changes=tuple(change.after for change in state.changes if change.is_material),
            lookup_failures=state.lookup_failures,
            new_paper_ids=state.new_paper_ids,
            unhandled=plan.unhandled if plan else (),
            failures=tuple(failures),
            review=state.review,
            superseded_conclusion_ids=(
                tuple(record.conclusion_id for record in live_before)
                if state.mode is MaintenanceMode.FULL_RERUN
                else ()
            ),
            regenerated_topic_id=regenerated_topic.topic_id if regenerated_topic else None,
            usage=self._usage_for(run_id, (completed - state.started_at).total_seconds() * 1000),
            dependency_graph_version=graph_version,
            started_at=state.started_at,
            completed_at=completed,
        )
        await self._diffs.save(diff)
        logger.info(
            "maintenance_completed",
            topic_id=topic.topic_id,
            run_id=run_id,
            mode=state.mode.value,
            changes=len(diff.changes),
            counts=diff.counts,
            failures=len(diff.failures),
        )
        if self._event_bus is not None:
            await self._event_bus.emit(
                EventType.MAINTENANCE_COMPLETED,
                MaintenancePayload(
                    topic_id=topic.topic_id,
                    diff_id=diff.id,
                    mode=state.mode.value,
                    changes=len(diff.changes),
                    affected=diff.affected_conclusions,
                    total=diff.total_conclusions,
                ),
                run_id=run_id,
                source=self.name,
            )
        return diff

    async def _record_new(
        self, state: MaintenanceState, target: MaintainedTopic
    ) -> tuple[list[ConclusionChange], list[MaintenanceFailure]]:
        changes: list[ConclusionChange] = []
        failures: list[MaintenanceFailure] = []
        verifications = {item.finding_id: item for item in state.new_verifications}
        new_papers = set(state.new_paper_ids)
        for finding in state.new_findings:
            record = await self._recorder.first_version(
                finding,
                topic_id=target.topic_id,
                run_id=state.run_id,
                verification=verifications.get(finding.id),
                change_type=ChangeType.NEW,
            )
            if record is None:
                failures.append(
                    MaintenanceFailure(
                        stage="record_new",
                        code="unresolvable_support",
                        message=f"finding {finding.id} has no resolvable support",
                    )
                )
                continue
            triggers: tuple[ChangeTrigger, ...]
            if state.mode is MaintenanceMode.FULL_RERUN:
                triggers = (
                    ChangeTrigger(
                        kind=TriggerKind.REGENERATION,
                        detail=f"regenerated by full rerun {state.run_id}",
                    ),
                )
            else:
                cited_new = [pid for pid in record.paper_ids if pid in new_papers]
                if not cited_new:
                    continue  # restates the old corpus: not a conclusion new papers produced
                triggers = tuple(
                    ChangeTrigger(
                        kind=TriggerKind.NEW_PAPER,
                        detail=f"new conclusion citing newly accepted paper {pid}",
                        paper_id=pid,
                        doi=next(
                            (link.doi for link in record.supports if link.paper_id == pid), None
                        ),
                    )
                    for pid in cited_new
                )
            try:
                await self._conclusions.append(record)
            except RepositoryError as exc:
                failures.append(
                    MaintenanceFailure(
                        stage="record_new",
                        code=exc.code,
                        message=exc.message,
                        conclusion_id=record.conclusion_id,
                    )
                )
                continue
            change = ConclusionChange(
                conclusion_id=record.conclusion_id,
                statement=record.statement,
                after_version=1,
                change_type=ChangeType.NEW,
                triggers=triggers,
                added_evidence_ids=record.evidence_ids,
                verification=record.verification,
            )
            changes.append(change)
            await self._emit_change(target.topic_id, change)
        return changes, failures

    async def _write_graph(
        self,
        topic_id: str,
        records: tuple[ConclusionRecord, ...],
        baseline: Mapping[str, IntegrityObservation],
    ) -> tuple[str | None, MaintenanceFailure | None]:
        papers = {pid for record in records for pid in record.paper_ids}
        graph = self._mapper.build(
            topic_id, records, {pid: item for pid, item in baseline.items() if pid in papers}
        )
        try:
            return await self._graph_store.write(graph), None
        except GraphStoreError as exc:
            # The graph is derived; the conclusions are already written.
            return None, MaintenanceFailure(
                stage="dependency_graph", code=exc.code, message=exc.message
            )

    async def _object(self, paper_id: str, object_id: str) -> KnowledgeObject | None:
        stored = await self._knowledge.get(paper_id)
        return stored.value.by_id(object_id) if stored is not None else None

    def _usage_for(self, run_id: str, latency_ms: float) -> MaintenanceUsage:
        if self._usage is None:
            return MaintenanceUsage(latency_ms=round(latency_ms, 3))
        summary = self._usage.summary(run_id)
        return MaintenanceUsage(
            llm_calls=len(summary.calls),
            prompt_tokens=summary.prompt_tokens,
            completion_tokens=summary.completion_tokens,
            total_tokens=summary.total_tokens,
            # No calls means nothing was spent, which is known; a priced-out call is not.
            estimated_cost_usd=summary.estimated_cost_usd if summary.calls else 0.0,
            latency_ms=round(latency_ms, 3),
        )

    async def _emit_change(self, topic_id: str, change: ConclusionChange) -> None:
        if self._event_bus is None:
            return
        await self._event_bus.emit(
            EventType.CONCLUSION_CHANGED,
            ConclusionChangePayload(
                topic_id=topic_id,
                conclusion_id=change.conclusion_id,
                change_type=change.change_type.value,
                after_version=change.after_version,
            ),
            source=self.name,
        )


def _require(topic: MaintainedTopic | None) -> MaintainedTopic:
    if topic is None:
        raise TopicNotFoundError("Maintenance state carries no topic")
    return topic
