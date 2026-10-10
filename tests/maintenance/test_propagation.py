"""Propagation rules, one at a time and combined.

Every rule in ``services/maintenance/propagation.py`` is exercised here with real domain
objects and the real contradiction detector; only the verifier's verdict is supplied by
hand, because producing it is the workflow's job, not the engine's.
"""

from __future__ import annotations

import pytest

from researchagent.config.schemas import PropagationSettings
from researchagent.core.validation import Confidence, ConfidenceSignal
from researchagent.models.evidence import EvidenceRole
from researchagent.models.integrity import IntegrityChange, IntegrityStatus, ObservationBasis
from researchagent.models.knowledge import KnowledgeObject
from researchagent.models.maintenance import (
    ChangeType,
    ConclusionRecord,
    ConclusionState,
    TriggerKind,
)
from researchagent.models.reasoning import Citation, VerificationResult, VerificationVerdict
from researchagent.services.maintenance.dependency_graph import DependencyGraphMapper
from researchagent.services.maintenance.propagation import (
    PropagationEngine,
    PropagationInput,
    ReverificationRequest,
)
from tests.maintenance.conftest import (
    TOPIC,
    conclusion,
    method_object,
    observation,
    paper,
    result_object,
    support,
)

P1, P2, P3, NEW = paper("p1"), paper("p2"), paper("p3"), paper("new")


def engine(**overrides: object) -> PropagationEngine:
    return PropagationEngine(PropagationSettings.model_validate(overrides))


def changed(paper_id: str, status: IntegrityStatus, **kwargs: object) -> IntegrityChange:
    return IntegrityChange(
        paper_id=paper_id, before=None, after=observation(paper_id, status, **kwargs)
    )  # type: ignore[arg-type]


def inputs(
    *records: ConclusionRecord,
    changes: tuple[IntegrityChange, ...] = (),
    new_objects: tuple[KnowledgeObject, ...] = (),
    supporting: tuple[KnowledgeObject, ...] = (),
) -> PropagationInput:
    graph = DependencyGraphMapper(schema_version="racm1").build(TOPIC, records)
    return PropagationInput(
        records=records,
        graph=graph,
        changes=changes,
        effective={change.paper_id: change.after for change in changes},
        new_objects=new_objects,
        supporting_objects={obj.id: obj for obj in supporting},
    )


def verdict(
    request: ReverificationRequest, kind: VerificationVerdict, score: float = 0.9
) -> VerificationResult:
    cited = (Citation(bundle_id="B-rv", evidence_ids=(request.remaining[0].evidence_id,)),)
    return VerificationResult(
        finding_id="RV",
        verdict=kind,
        supporting=cited if kind is VerificationVerdict.VERIFIED else (),
        confidence=Confidence(
            score=score,
            signals=(
                ConfidenceSignal(name="provenance_resolved", value=score, observation="test"),
            ),
        ),
        verified_by="verification",
    )


# --- integrity rules ----------------------------------------------------------------------


def test_sole_support_retracted_invalidates() -> None:
    record = conclusion("C1", support(P1), support(P1, 1))
    plan = engine().plan(inputs(record, changes=(changed(P1, IntegrityStatus.RETRACTED),)))

    assert plan.reverify == ()
    (outcome,) = plan.outcomes
    assert outcome.change_type is ChangeType.INVALIDATED

    updated, change = engine().apply(outcome, run_id="M-1")
    assert updated.state is ConclusionState.INVALIDATED
    assert updated.supports == () and updated.version == 2
    assert set(change.removed_evidence_ids) == set(record.evidence_ids)
    (trigger,) = change.triggers
    assert trigger.kind is TriggerKind.INTEGRITY
    assert trigger.status is IntegrityStatus.RETRACTED
    assert trigger.doi == "10.5555/p1"
    assert trigger.registry == "crossref"
    assert trigger.notice_doi == "10.5555/notice-p1"
    assert trigger.notice_date == "2026-09-01"


def test_withdrawal_invalidates_like_retraction() -> None:
    record = conclusion("C1", support(P1))
    plan = engine().plan(inputs(record, changes=(changed(P1, IntegrityStatus.WITHDRAWN),)))
    assert plan.outcomes[0].change_type is ChangeType.INVALIDATED


def test_one_retracted_one_valid_is_not_invalidated() -> None:
    record = conclusion("C1", support(P1), support(P2))
    plan = engine().plan(inputs(record, changes=(changed(P1, IntegrityStatus.RETRACTED),)))

    assert plan.outcomes == ()
    (request,) = plan.reverify
    assert [link.paper_id for link in request.remaining] == [P2]
    assert [item.link.paper_id for item in request.excluded] == [P1]

    outcome = engine().resolve(
        request, verdict(request, VerificationVerdict.VERIFIED), run_id="M-1"
    )
    assert outcome.change_type is not ChangeType.INVALIDATED
    # One of two papers left: source diversity fell below the configured minimum of 2.
    assert outcome.change_type is ChangeType.WEAKENED
    assert any("source diversity" in reason for reason in outcome.reasons)


def test_reverified_with_enough_support_left_is_kept() -> None:
    record = conclusion("C1", support(P1), support(P2), support(P3))
    request = (
        engine().plan(inputs(record, changes=(changed(P1, IntegrityStatus.RETRACTED),))).reverify[0]
    )

    outcome = engine().resolve(
        request, verdict(request, VerificationVerdict.VERIFIED), run_id="M-1"
    )
    updated, change = engine().apply(outcome, run_id="M-1")

    assert outcome.change_type is ChangeType.KEPT
    assert updated.state is ConclusionState.ACTIVE
    assert updated.paper_ids == (P2, P3)
    assert change.removed_evidence_ids == ("E-p1-0",)
    assert change.verification is not None
    assert {trigger.kind for trigger in change.triggers} == {
        TriggerKind.INTEGRITY,
        TriggerKind.REVERIFICATION,
    }


@pytest.mark.parametrize(
    ("kind", "score", "reason"),
    [
        (VerificationVerdict.PARTIALLY_SUPPORTED, 0.9, "verdict is partially_supported"),
        (VerificationVerdict.INSUFFICIENT_EVIDENCE, 0.9, "verdict is insufficient_evidence"),
        (VerificationVerdict.VERIFIED, 0.5, "confidence fell by 0.40"),
    ],
)
def test_reverification_weakens(kind: VerificationVerdict, score: float, reason: str) -> None:
    record = conclusion("C1", support(P1), support(P2), support(P3))
    request = (
        engine().plan(inputs(record, changes=(changed(P1, IntegrityStatus.RETRACTED),))).reverify[0]
    )

    outcome = engine().resolve(request, verdict(request, kind, score), run_id="M-1")

    assert outcome.change_type is ChangeType.WEAKENED
    assert any(reason in item for item in outcome.reasons)


def test_too_little_support_retained_weakens() -> None:
    record = conclusion("C1", support(P1), support(P1, 1), support(P1, 2), support(P2), support(P3))
    request = (
        engine().plan(inputs(record, changes=(changed(P1, IntegrityStatus.RETRACTED),))).reverify[0]
    )

    outcome = engine().resolve(
        request, verdict(request, VerificationVerdict.VERIFIED), run_id="M-1"
    )

    assert outcome.change_type is ChangeType.WEAKENED
    assert any("support retained 0.40" in reason for reason in outcome.reasons)


def test_failed_reverification_needs_review_not_kept() -> None:
    record = conclusion("C1", support(P1), support(P2), support(P3))
    request = (
        engine().plan(inputs(record, changes=(changed(P1, IntegrityStatus.RETRACTED),))).reverify[0]
    )

    outcome = engine().resolve(request, None, run_id="M-1", failure="agent_execution_error")

    assert outcome.change_type is ChangeType.NEEDS_REVIEW


@pytest.mark.parametrize(
    "status", [IntegrityStatus.EXPRESSION_OF_CONCERN, IntegrityStatus.CORRECTED]
)
def test_concern_or_correction_flags_for_review_never_invalidates(status: IntegrityStatus) -> None:
    record = conclusion("C1", support(P1))
    plan = engine().plan(inputs(record, changes=(changed(P1, status),)))

    (outcome,) = plan.outcomes
    assert outcome.change_type is ChangeType.NEEDS_REVIEW
    assert plan.reverify == ()
    updated, change = engine().apply(outcome, run_id="M-1")
    assert updated.state is ConclusionState.NEEDS_REVIEW
    assert updated.supports == record.supports  # nothing removed
    assert change.removed_evidence_ids == ()
    assert change.triggers[0].status is status


def test_pdf_notice_flags_for_review() -> None:
    record = conclusion("C1", support(P1))
    flagged = changed(P1, IntegrityStatus.UNKNOWN, basis=ObservationBasis.PDF_NOTICE)
    plan = engine().plan(inputs(record, changes=(flagged,)))
    assert plan.outcomes[0].change_type is ChangeType.NEEDS_REVIEW


# --- affected set -------------------------------------------------------------------------


def test_only_dependent_conclusions_are_touched_or_reverified() -> None:
    hit = conclusion("C-hit", support(P1), support(P2), support(P3, 1))
    untouched = conclusion("C-untouched", support(P2, 5), support(P3))
    plan = engine().plan(inputs(hit, untouched, changes=(changed(P1, IntegrityStatus.RETRACTED),)))

    assert plan.affected_ids == ("C-hit",)
    assert [request.record.conclusion_id for request in plan.reverify] == ["C-hit"]
    assert all(outcome.record.conclusion_id != "C-untouched" for outcome in plan.outcomes)


def test_nothing_changed_gives_an_empty_plan() -> None:
    record = conclusion("C1", support(P1))
    current = changed(P1, IntegrityStatus.CURRENT)
    failed = changed(P1, IntegrityStatus.UNKNOWN)  # lookup failure: uninformative

    for changes in ((), (current,), (failed,)):
        plan = engine().plan(inputs(record, changes=changes))
        assert plan.outcomes == () and plan.reverify == () and plan.affected_ids == ()


def test_unruled_transition_is_reported_not_acted_on() -> None:
    record = conclusion("C1", support(P1))
    reinstated = IntegrityChange(
        paper_id=P1,
        before=observation(P1, IntegrityStatus.EXPRESSION_OF_CONCERN),
        after=observation(P1, IntegrityStatus.CURRENT),
    )
    plan = engine().plan(inputs(record, changes=(reinstated,)))

    assert plan.outcomes == () and plan.affected_ids == ()
    assert "no rule applies" in plan.unhandled[0]


def test_invalidated_conclusions_are_not_reconsidered() -> None:
    dead = conclusion("C1", support(P1))
    outcome = (
        engine().plan(inputs(dead, changes=(changed(P1, IntegrityStatus.RETRACTED),))).outcomes[0]
    )
    invalidated, _ = engine().apply(outcome, run_id="M-1")

    plan = engine().plan(
        inputs(invalidated, changes=(changed(P1, IntegrityStatus.EXPRESSION_OF_CONCERN),))
    )
    assert plan.affected_ids == ()


def test_propagation_disabled_detects_but_changes_nothing() -> None:
    record = conclusion("C1", support(P1))
    plan = engine(enabled=False).plan(
        inputs(record, changes=(changed(P1, IntegrityStatus.RETRACTED),))
    )

    assert plan.affected_ids == ("C1",)
    assert plan.outcomes == () and plan.reverify == ()


# --- new papers ---------------------------------------------------------------------------


def test_new_paper_contradiction_marks_newly_contradicted() -> None:
    old_result = result_object(P1, 0.82)
    link = support(P1).model_copy(update={"knowledge_object_id": old_result.id})
    record = conclusion("C1", link)
    newcomer = result_object(NEW, 0.41)

    plan = engine().plan(inputs(record, new_objects=(newcomer,), supporting=(old_result,)))

    (outcome,) = plan.outcomes
    assert outcome.change_type is ChangeType.NEWLY_CONTRADICTED
    (contradiction,) = outcome.contradictions
    # Both sides are carried, never resolved.
    assert {contradiction.left_paper_id, contradiction.right_paper_id} == {P1, NEW}
    assert contradiction.left_evidence and contradiction.right_evidence
    updated, change = engine().apply(outcome, run_id="M-1")
    assert updated.state is ConclusionState.CONTRADICTED
    assert updated.supports == record.supports
    assert change.triggers[0].kind is TriggerKind.CONTRADICTION
    assert change.triggers[0].contradiction_id == contradiction.id


def test_agreeing_new_paper_is_ignored_by_the_contradiction_rule() -> None:
    old_result = result_object(P1, 0.82)
    record = conclusion("C1", support(P1).model_copy(update={"knowledge_object_id": old_result.id}))
    plan = engine().plan(
        inputs(record, new_objects=(result_object(NEW, 0.82),), supporting=(old_result,))
    )
    assert plan.affected_ids == ()  # results are not in the agreement kinds either


def test_new_supporting_paper_is_kept_with_support_added() -> None:
    old_method = method_object(P1, "Circuit Breaker")
    record = conclusion("C1", support(P1))
    newcomer = method_object(NEW, "circuit  breaker")  # same entity, different spacing/case

    plan = engine().plan(inputs(record, new_objects=(newcomer,), supporting=(old_method,)))

    (outcome,) = plan.outcomes
    assert outcome.change_type is ChangeType.KEPT
    updated, change = engine().apply(outcome, run_id="M-1")
    assert updated.paper_ids == (P1, NEW)
    assert change.added_evidence_ids == (newcomer.evidence[0].id,)
    added = next(link for link in updated.supports if link.paper_id == NEW)
    assert added.role is EvidenceRole.CORROBORATING
    assert change.triggers[0].kind is TriggerKind.NEW_PAPER


# --- precedence and history ---------------------------------------------------------------


def test_several_rules_pick_by_precedence_and_keep_every_trigger() -> None:
    record = conclusion("C1", support(P1), support(P2), support(P3))
    plan = engine().plan(
        inputs(
            record,
            changes=(
                changed(P1, IntegrityStatus.RETRACTED),
                changed(P2, IntegrityStatus.EXPRESSION_OF_CONCERN),
            ),
        )
    )
    (request,) = plan.reverify
    assert request.also == (ChangeType.NEEDS_REVIEW,)

    outcome = engine().resolve(
        request, verdict(request, VerificationVerdict.VERIFIED), run_id="M-1"
    )

    assert outcome.change_type is ChangeType.NEEDS_REVIEW  # needs_review outranks kept
    statuses = {trigger.status for trigger in outcome.triggers if trigger.status}
    assert statuses == {IntegrityStatus.RETRACTED, IntegrityStatus.EXPRESSION_OF_CONCERN}


def test_kept_does_not_clear_an_earlier_flag() -> None:
    old_method = method_object(P1, "Circuit Breaker")
    flagged = conclusion("C1", support(P1)).model_copy(
        update={"state": ConclusionState.NEEDS_REVIEW}
    )
    outcome = (
        engine()
        .plan(
            inputs(
                flagged,
                new_objects=(method_object(NEW, "Circuit Breaker"),),
                supporting=(old_method,),
            )
        )
        .outcomes[0]
    )

    updated, _ = engine().apply(outcome, run_id="M-1")

    assert outcome.change_type is ChangeType.KEPT
    assert updated.state is ConclusionState.NEEDS_REVIEW
