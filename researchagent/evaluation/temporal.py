"""Temporal replay scaffold: freeze discovery at T0, add papers up to T1, emit the diff.

Ships the mechanism and the gold-annotation *schema* only. Annotations — the expected
change type of each T0 conclusion once T1's papers are known — must come from a person
reading the papers; none are provided, and a template with an empty list is the only file.
Papers carry a publication year, not a date, so windows are whole years.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel, Field, model_validator

from researchagent.core.exceptions import ConfigurationError
from researchagent.evaluation.gold import ReviewStatus
from researchagent.models.maintenance import ChangeType, ConclusionDiff
from researchagent.models.paper import Paper


class ReplayWindow(BaseModel):
    model_config = {"frozen": True}

    frozen_through_year: int = Field(ge=1500, le=2200, description="T0")
    added_through_year: int = Field(ge=1500, le=2200, description="T1")

    @model_validator(mode="after")
    def _ordered(self) -> ReplayWindow:
        if self.added_through_year <= self.frozen_through_year:
            raise ValueError("T1 must be after T0")
        return self


class CorpusSplit(BaseModel):
    model_config = {"frozen": True}

    frozen: tuple[str, ...] = ()
    added: tuple[str, ...] = ()
    after_window: tuple[str, ...] = ()
    # No year: cannot be placed on either side, so used on neither.
    undated: tuple[str, ...] = ()


def split_corpus(papers: Iterable[Paper], window: ReplayWindow) -> CorpusSplit:
    frozen: list[str] = []
    added: list[str] = []
    later: list[str] = []
    undated: list[str] = []
    for paper in papers:
        if paper.year is None:
            undated.append(paper.id)
        elif paper.year <= window.frozen_through_year:
            frozen.append(paper.id)
        elif paper.year <= window.added_through_year:
            added.append(paper.id)
        else:
            later.append(paper.id)
    return CorpusSplit(
        frozen=tuple(sorted(frozen)),
        added=tuple(sorted(added)),
        after_window=tuple(sorted(later)),
        undated=tuple(sorted(undated)),
    )


class TemporalAnnotation(BaseModel):
    """What a person expects to happen to one T0 conclusion by T1."""

    model_config = {"frozen": True}

    conclusion_id: str = Field(min_length=1)
    # None means "should be unchanged".
    expected_change: ChangeType | None = None
    rationale: str = Field(min_length=1)
    annotator: str = Field(min_length=1)
    status: ReviewStatus = ReviewStatus.DRAFT


class TemporalGoldSet(BaseModel):
    model_config = {"frozen": True}

    version: str = Field(min_length=1)
    topic_id: str = Field(min_length=1)
    window: ReplayWindow
    annotations: tuple[TemporalAnnotation, ...] = ()

    @classmethod
    def load(cls, path: Path) -> TemporalGoldSet:
        if not path.is_file():
            raise ConfigurationError("Temporal gold set not found", file=str(path))
        try:
            return cls.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ConfigurationError(
                "Temporal gold set could not be read", file=str(path), reason=str(exc)
            ) from exc


class TemporalScore(BaseModel):
    """Agreement with *reviewed* annotations; drafts are counted, never scored."""

    model_config = {"frozen": True}

    reviewed: int
    drafts: int
    matched: int
    accuracy: float | None
    # expected -> predicted -> count; "unchanged" stands for None.
    confusion: dict[str, dict[str, int]] = Field(default_factory=dict)


def score_replay(gold: TemporalGoldSet, diff: ConclusionDiff) -> TemporalScore:
    reviewed = [item for item in gold.annotations if item.status is ReviewStatus.REVIEWED]
    confusion: dict[str, dict[str, int]] = {}
    matched = 0
    for item in reviewed:
        change = diff.change_for(item.conclusion_id)
        predicted = change.change_type if change else None
        matched += predicted is item.expected_change
        row = confusion.setdefault(_label(item.expected_change), {})
        row[_label(predicted)] = row.get(_label(predicted), 0) + 1
    return TemporalScore(
        reviewed=len(reviewed),
        drafts=len(gold.annotations) - len(reviewed),
        matched=matched,
        accuracy=round(matched / len(reviewed), 6) if reviewed else None,
        confusion=confusion,
    )


def _label(kind: ChangeType | None) -> str:
    return kind.value if kind is not None else "unchanged"
