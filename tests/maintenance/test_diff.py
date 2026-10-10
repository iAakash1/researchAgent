"""The diff contract: no change without a trigger, NEW has no past, counts always line up."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from researchagent.models.integrity import IntegrityStatus
from researchagent.models.maintenance import (
    ChangeTrigger,
    ChangeType,
    ConclusionChange,
    ConclusionDiff,
    MaintenanceMode,
    TriggerKind,
)
from tests.maintenance.conftest import observation, paper

TRIGGER = ChangeTrigger.from_observation(observation(paper("p1"), IntegrityStatus.RETRACTED))


def test_a_change_must_name_what_caused_it() -> None:
    with pytest.raises(ValidationError):
        ConclusionChange(
            conclusion_id="C1",
            before_version=1,
            after_version=2,
            change_type=ChangeType.INVALIDATED,
            triggers=(),
        )


def test_new_has_no_previous_version_and_others_must() -> None:
    with pytest.raises(ValidationError):
        ConclusionChange(
            conclusion_id="C1",
            before_version=1,
            after_version=1,
            change_type=ChangeType.NEW,
            triggers=(TRIGGER,),
        )
    with pytest.raises(ValidationError):
        ConclusionChange(
            conclusion_id="C1", after_version=2, change_type=ChangeType.KEPT, triggers=(TRIGGER,)
        )


def test_counts_cover_every_type_and_survive_serialisation() -> None:
    diff = ConclusionDiff(
        topic_id="T",
        maintenance_run_id="M",
        mode=MaintenanceMode.INCREMENTAL,
        changes=(
            ConclusionChange(
                conclusion_id="C1",
                before_version=1,
                after_version=2,
                change_type=ChangeType.INVALIDATED,
                triggers=(TRIGGER,),
            ),
            ConclusionChange(
                conclusion_id="C2",
                after_version=1,
                change_type=ChangeType.NEW,
                triggers=(ChangeTrigger(kind=TriggerKind.NEW_PAPER, detail="new paper"),),
            ),
        ),
    )

    assert diff.counts == {
        "kept": 0,
        "weakened": 0,
        "invalidated": 1,
        "needs_review": 0,
        "newly_contradicted": 0,
        "new": 1,
    }
    dumped = diff.model_dump(mode="json")
    assert dumped["counts"]["invalidated"] == 1
    assert ConclusionDiff.model_validate_json(diff.model_dump_json()) == diff
    assert dumped["changes"][0]["triggers"][0]["notice_doi"] == "10.5555/notice-p1"


def test_trigger_carries_the_registry_observation() -> None:
    assert TRIGGER.kind is TriggerKind.INTEGRITY
    assert (TRIGGER.registry, TRIGGER.status, TRIGGER.doi) == (
        "crossref",
        IntegrityStatus.RETRACTED,
        "10.5555/p1",
    )
    assert TRIGGER.observation_id is not None
