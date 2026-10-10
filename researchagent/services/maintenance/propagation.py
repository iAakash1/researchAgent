"""Deterministic propagation of integrity changes and new papers to conclusions.

No language model decides anything here. The engine:

1. takes the papers whose integrity *materially* changed, and finds the conclusions that
   depend on them by reverse traversal of the dependency graph — every other conclusion is
   left untouched and is never re-verified;
2. applies the configured rules to each affected conclusion:

   * evidence from a paper with an excluded status (retracted, withdrawn) stops counting:
     no support left → INVALIDATED; support left → re-verify on what remains →
     KEPT or WEAKENED (``resolve``);
   * a supporting paper newly under an expression of concern or corrected →
     NEEDS_REVIEW, never invalidated;
   * a new paper whose knowledge makes the existing ``ContradictionDetector`` fire against
     a supporting knowledge object → NEWLY_CONTRADICTED, both sides carried;
   * a new paper naming the same entity as a supporting knowledge object, without a
     contradiction → KEPT, with the corroborating support added and recorded;

3. when several rules fire for one conclusion, picks the change type by the configured
   precedence and keeps every trigger.

Re-verification itself is an agent call, so it happens in the workflow: ``plan`` returns
requests, the workflow runs the existing verification agent, and ``resolve`` turns the
verdict into an outcome. ``apply`` then appends the next version — history is never edited.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime

from researchagent.config.schemas import PropagationSettings
from researchagent.core.logging import get_logger
from researchagent.models.bundle import Contradiction
from researchagent.models.evidence import EvidenceRole
from researchagent.models.integrity import IntegrityChange
from researchagent.models.maintenance import (
    ChangeTrigger,
    ChangeType,
    ConclusionChange,
    ConclusionRecord,
    ExcludedSupport,
    SupportLink,
    TriggerKind,
    VerificationSnapshot,
)
from researchagent.models.reasoning import VerificationResult, VerificationVerdict
from researchagent.schemas.maintenance import (
    ConclusionOutcome,
    PropagationInput,
    PropagationPlan,
    ReverificationRequest,
)
from researchagent.services.evidence.contradictions import ContradictionDetector
from researchagent.services.maintenance.dependency_graph import dependents_of
from researchagent.services.maintenance.recorder import snapshot
from researchagent.utils.text import normalise

logger = get_logger(__name__)


class _Hits:
    """Rule hits for one conclusion, accumulated before a decision is made."""

    def __init__(self) -> None:
        self.excluded: dict[str, ExcludedSupport] = {}
        self.review_flags: list[str] = []
        self.contradictions: dict[str, Contradiction] = {}
        self.added: dict[str, SupportLink] = {}
        self.triggers: dict[str, ChangeTrigger] = {}

    def trigger(self, trigger: ChangeTrigger) -> None:
        key = "|".join(
            str(part)
            for part in (
                trigger.kind,
                trigger.observation_id,
                trigger.contradiction_id,
                trigger.paper_id,
            )
        )
        self.triggers.setdefault(key, trigger)

    @property
    def fired(self) -> bool:
        return bool(self.excluded or self.review_flags or self.contradictions or self.added)


class PropagationEngine:
    name = "propagation_engine"

    def __init__(
        self, settings: PropagationSettings, detector: ContradictionDetector | None = None
    ) -> None:
        self._settings = settings
        self._detector = detector or ContradictionDetector()
        self._rank = {kind: index for index, kind in enumerate(settings.precedence)}

    def with_settings(self, settings: PropagationSettings) -> PropagationEngine:
        return PropagationEngine(settings, self._detector)

    def plan(self, data: PropagationInput) -> PropagationPlan:
        live = {record.conclusion_id: record for record in data.records if record.is_live}
        changed = {change.paper_id: change for change in data.changes if change.is_material}
        reached = {
            conclusion_id: papers
            for conclusion_id, papers in dependents_of(data.graph, changed).items()
            if conclusion_id in live
        }
        hits: dict[str, _Hits] = defaultdict(_Hits)
        unhandled: list[str] = []

        for conclusion_id, papers in sorted(reached.items()):
            self._integrity_hits(live[conclusion_id], papers, changed, data, hits, unhandled)
        if data.new_objects:
            self._new_paper_hits(live, data, hits)

        affected = tuple(sorted(conclusion_id for conclusion_id, hit in hits.items() if hit.fired))
        if not self._settings.enabled:
            logger.info("propagation_disabled", affected=len(affected))
            return PropagationPlan(affected_ids=affected, unhandled=tuple(unhandled))

        outcomes: list[ConclusionOutcome] = []
        requests: list[ReverificationRequest] = []
        for conclusion_id in affected:
            decided = self._decide(live[conclusion_id], hits[conclusion_id])
            if isinstance(decided, ReverificationRequest):
                requests.append(decided)
            else:
                outcomes.append(decided)

        logger.info(
            "propagation_planned",
            changed_papers=len(changed),
            affected=len(affected),
            decided=len(outcomes),
            reverify=len(requests),
            unhandled=len(unhandled),
        )
        return PropagationPlan(
            outcomes=tuple(outcomes),
            reverify=tuple(requests),
            affected_ids=affected,
            unhandled=tuple(unhandled),
        )

    def resolve(
        self,
        request: ReverificationRequest,
        verification: VerificationResult | None,
        *,
        run_id: str,
        failure: str | None = None,
    ) -> ConclusionOutcome:
        """KEPT or WEAKENED from the surviving support and the verifier's new verdict."""
        record = request.record
        thresholds = self._settings.weakened
        excluded_ids = {item.link.evidence_id for item in request.excluded}
        surviving = [link for link in record.supports if link.evidence_id not in excluded_ids]
        retained = len(surviving) / len(record.supports) if record.supports else 0.0
        papers_before = len(record.paper_ids)
        papers_after = len({link.paper_id for link in request.remaining})

        reasons: list[str] = []
        if retained < thresholds.min_support_retained:
            reasons.append(
                f"support retained {retained:.2f} is below {thresholds.min_support_retained:.2f}"
            )
        if papers_after < thresholds.min_supporting_papers and papers_after < papers_before:
            reasons.append(
                f"source diversity fell from {papers_before} to {papers_after} paper(s) "
                f"(minimum {thresholds.min_supporting_papers})"
            )

        verified = snapshot(verification, run_id)
        if failure is not None:
            base = ChangeType.NEEDS_REVIEW
            detail = f"re-verification could not run: {failure}"
            reasons.append(detail)
        elif verification is None:
            base = ChangeType.WEAKENED if reasons else ChangeType.KEPT
            detail = "re-verification disabled; decided on support thresholds only"
        else:
            if verification.verdict is not VerificationVerdict.VERIFIED:
                reasons.append(f"re-verification verdict is {verification.verdict.value}")
            drop = _confidence_drop(record.verification, verified)
            if drop is not None and drop > thresholds.max_confidence_drop:
                reasons.append(
                    f"verification confidence fell by {drop:.2f} "
                    f"(tolerance {thresholds.max_confidence_drop:.2f})"
                )
            base = ChangeType.WEAKENED if reasons else ChangeType.KEPT
            detail = (
                f"re-verified on {len(request.remaining)} remaining evidence item(s): "
                f"{verification.verdict.value}, confidence {verification.confidence.score:.2f}"
            )

        return ConclusionOutcome(
            record=record,
            change_type=self._pick({base, *request.also}),
            supports=request.remaining,
            excluded=request.excluded,
            added=request.added,
            contradictions=request.contradictions,
            review_flags=request.review_flags,
            triggers=(
                *request.triggers,
                ChangeTrigger(kind=TriggerKind.REVERIFICATION, detail=detail),
            ),
            verification=verified,
            reasons=tuple(reasons),
        )

    def apply(
        self, outcome: ConclusionOutcome, *, run_id: str, now: datetime | None = None
    ) -> tuple[ConclusionRecord, ConclusionChange]:
        """The next version, and the change that explains it."""
        previous = outcome.record
        kind = outcome.change_type
        # KEPT confirms a conclusion; it does not clear an earlier flag on it.
        state = previous.state if kind is ChangeType.KEPT else kind.resulting_state
        supports = () if kind is ChangeType.INVALIDATED else outcome.supports
        record = ConclusionRecord(
            conclusion_id=previous.conclusion_id,
            version=previous.version + 1,
            previous_version=previous.version,
            topic_id=previous.topic_id,
            run_id=run_id,
            question_id=previous.question_id,
            statement=previous.statement,
            state=state,
            change_type=kind,
            supports=supports,
            excluded=(*previous.excluded, *outcome.excluded),
            contradictions=_unique_contradictions(
                (*previous.contradictions, *outcome.contradictions)
            ),
            review_flags=tuple(dict.fromkeys((*previous.review_flags, *outcome.review_flags))),
            unresolved_evidence_ids=previous.unresolved_evidence_ids,
            verification=outcome.verification or previous.verification,
            source_finding_id=previous.source_finding_id,
            recorded_at=now or datetime.now(UTC),
        )
        before_ids = set(previous.evidence_ids)
        after_ids = set(record.evidence_ids)
        change = ConclusionChange(
            conclusion_id=previous.conclusion_id,
            statement=previous.statement,
            before_version=previous.version,
            after_version=record.version,
            change_type=kind,
            triggers=outcome.triggers,
            removed_evidence_ids=tuple(sorted(before_ids - after_ids)),
            added_evidence_ids=tuple(sorted(after_ids - before_ids)),
            verification=outcome.verification,
            contradictions=outcome.contradictions,
            reasons=outcome.reasons,
        )
        return record, change

    # --- rule hits ----------------------------------------------------------------------

    def _integrity_hits(
        self,
        record: ConclusionRecord,
        reaching: frozenset[str],
        changed: Mapping[str, IntegrityChange],
        data: PropagationInput,
        hits: dict[str, _Hits],
        unhandled: list[str],
    ) -> None:
        excluded_statuses = set(self._settings.excluded_statuses)
        review_statuses = set(self._settings.review_statuses)
        hit = hits[record.conclusion_id]

        # Exclusion applies to every supporting paper whose effective status excludes it,
        # not only to the one that changed: support never outlives its paper's retraction.
        for link in record.supports:
            current = changed[link.paper_id].after if link.paper_id in changed else None
            current = current or data.effective.get(link.paper_id)
            if current is not None and current.status in excluded_statuses:
                hit.excluded[link.evidence_id] = ExcludedSupport(
                    link=link, status=current.status, observation_id=current.id
                )
                hit.trigger(ChangeTrigger.from_observation(current))

        for paper_id in sorted(reaching):
            after = changed[paper_id].after
            if after.status in excluded_statuses:
                continue
            if after.status in review_statuses or after.flags_review:
                hit.review_flags.append(after.id)
                hit.trigger(ChangeTrigger.from_observation(after))
            else:
                before = changed[paper_id].before
                unhandled.append(
                    f"{record.conclusion_id}: {paper_id} moved "
                    f"{before.status.value if before else 'unobserved'} -> {after.status.value}; "
                    "no rule applies"
                )

    def _new_paper_hits(
        self,
        live: Mapping[str, ConclusionRecord],
        data: PropagationInput,
        hits: dict[str, _Hits],
    ) -> None:
        by_object: dict[str, list[ConclusionRecord]] = defaultdict(list)
        for record in live.values():
            for object_id in record.knowledge_object_ids:
                by_object[object_id].append(record)
        existing = {
            object_id: obj
            for object_id, obj in data.supporting_objects.items()
            if object_id in by_object
        }
        new_ids = {obj.id for obj in data.new_objects}
        contradictions = self._detector.detect((*existing.values(), *data.new_objects))

        disagreeing: set[tuple[str, str]] = set()
        for contradiction in contradictions:
            sides = {contradiction.left_object_id, contradiction.right_object_id}
            old_side = sides - new_ids
            new_side = sides & new_ids
            if len(old_side) != 1 or len(new_side) != 1:
                continue  # old-vs-old was already known; new-vs-new is not about a conclusion
            old_id, new_id = next(iter(old_side)), next(iter(new_side))
            disagreeing.add((old_id, new_id))
            new_paper = (
                contradiction.left_paper_id
                if contradiction.left_object_id == new_id
                else contradiction.right_paper_id
            )
            for record in by_object.get(old_id, ()):
                hit = hits[record.conclusion_id]
                hit.contradictions[contradiction.id] = contradiction
                hit.trigger(
                    ChangeTrigger(
                        kind=TriggerKind.CONTRADICTION,
                        detail=contradiction.description,
                        paper_id=new_paper,
                        doi=data.dois.get(new_paper),
                        contradiction_id=contradiction.id,
                    )
                )

        if not self._settings.agreement.enabled:
            return
        kinds = set(self._settings.agreement.kinds)
        by_name: dict[tuple[str, str], list[str]] = defaultdict(list)
        for object_id, obj in existing.items():
            if obj.kind in kinds:
                by_name[(obj.kind.value, normalise(obj.name))].append(object_id)
        for new in data.new_objects:
            if new.kind not in kinds:
                continue
            for old_id in by_name.get((new.kind.value, normalise(new.name)), ()):
                if (old_id, new.id) in disagreeing:
                    continue
                for record in by_object[old_id]:
                    if new.paper_id in record.paper_ids:
                        continue
                    hit = hits[record.conclusion_id]
                    for evidence in new.evidence:
                        hit.added.setdefault(
                            evidence.id,
                            SupportLink(
                                evidence_id=evidence.id,
                                knowledge_object_id=new.id,
                                paper_id=new.paper_id,
                                location=evidence.location,
                                doi=data.dois.get(new.paper_id),
                                quote=evidence.quote or "",
                                role=EvidenceRole.CORROBORATING,
                            ),
                        )
                    hit.trigger(
                        ChangeTrigger(
                            kind=TriggerKind.NEW_PAPER,
                            detail=(
                                f"{new.paper_id} also names the {new.kind.value} {new.name!r} "
                                f"this conclusion rests on"
                            ),
                            paper_id=new.paper_id,
                            doi=data.dois.get(new.paper_id),
                        )
                    )

    # --- decision -----------------------------------------------------------------------

    def _decide(
        self, record: ConclusionRecord, hit: _Hits
    ) -> ConclusionOutcome | ReverificationRequest:
        also: set[ChangeType] = set()
        if hit.contradictions:
            also.add(ChangeType.NEWLY_CONTRADICTED)
        if hit.review_flags:
            also.add(ChangeType.NEEDS_REVIEW)
        if hit.added:
            also.add(ChangeType.KEPT)

        excluded = tuple(hit.excluded.values())
        added = tuple(hit.added.values())
        remaining = (
            *(link for link in record.supports if link.evidence_id not in hit.excluded),
            *added,
        )
        triggers = tuple(hit.triggers.values())
        contradictions = tuple(hit.contradictions.values())

        if excluded and not remaining:
            # Forced regardless of precedence: a conclusion with nothing under it cannot be
            # anything but invalidated.
            return ConclusionOutcome(
                record=record,
                change_type=ChangeType.INVALIDATED,
                excluded=excluded,
                contradictions=contradictions,
                review_flags=tuple(hit.review_flags),
                triggers=triggers,
                reasons=(f"all {len(record.supports)} supporting evidence item(s) excluded",),
            )
        if excluded:
            return ReverificationRequest(
                record=record,
                remaining=remaining,
                excluded=excluded,
                added=added,
                triggers=triggers,
                also=tuple(sorted(also, key=self._rank.__getitem__)),
                contradictions=contradictions,
                review_flags=tuple(hit.review_flags),
            )
        return ConclusionOutcome(
            record=record,
            change_type=self._pick(also),
            supports=remaining,
            added=added,
            contradictions=contradictions,
            review_flags=tuple(hit.review_flags),
            triggers=triggers,
        )

    def _pick(self, candidates: set[ChangeType]) -> ChangeType:
        return min(candidates, key=self._rank.__getitem__)


def _confidence_drop(
    before: VerificationSnapshot | None, after: VerificationSnapshot | None
) -> float | None:
    """How far the verifier's confidence fell, when both scores rest on observations."""
    if before is None or after is None:
        return None
    if not before.confidence.is_grounded or not after.confidence.is_grounded:
        return None
    return round(before.confidence.score - after.confidence.score, 6)


def _unique_contradictions(items: Sequence[Contradiction]) -> tuple[Contradiction, ...]:
    return tuple({item.id: item for item in items}.values())


__all__ = [
    "ConclusionOutcome",
    "PropagationEngine",
    "PropagationInput",
    "PropagationPlan",
    "ReverificationRequest",
]
