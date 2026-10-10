"""Turns a registry lookup into an integrity observation.

Deterministic and configuration-driven: ``config/integrity.yaml`` says what each update
type means and how statuses rank. Nothing here asks a model anything.
"""

from __future__ import annotations

from datetime import datetime

from researchagent.config.schemas import IntegrityConfig
from researchagent.core.interfaces.integrity import LookupOutcome, RegistryLookup
from researchagent.models.integrity import (
    IntegrityObservation,
    IntegrityStatus,
    ObservationBasis,
    UpdateNotice,
)


def normalise_update_type(raw: str) -> str:
    """``"Expression of concern"`` and ``expression-of-concern`` are one type."""
    return "_".join(raw.strip().lower().replace("-", " ").split())


class UpdateClassifier:
    """Maps a work's update notices onto one status, by configured meaning and severity."""

    def __init__(self, config: IntegrityConfig) -> None:
        self._config = config
        self._rank = {status: index for index, status in enumerate(config.severity)}
        self._ignored = {normalise_update_type(item) for item in config.ignored_update_types}
        self._meaning = {
            normalise_update_type(name): status for name, status in config.update_types.items()
        }

    def classify(
        self, lookup: RegistryLookup, *, paper_id: str, retrieved_at: datetime | None = None
    ) -> IntegrityObservation:
        identifier = f"doi:{lookup.doi}"
        stamp = retrieved_at or lookup.retrieved_at

        if lookup.outcome is not LookupOutcome.FOUND:
            return IntegrityObservation(
                paper_id=paper_id,
                identifier=identifier,
                doi=lookup.doi,
                status=IntegrityStatus.UNKNOWN,
                basis=ObservationBasis.LOOKUP_FAILED,
                registry=lookup.registry,
                observation=f"{lookup.outcome.value}: {lookup.detail}",
                retrieved_at=stamp,
            )

        relevant: list[tuple[IntegrityStatus, UpdateNotice]] = []
        unmapped: list[UpdateNotice] = []
        ignored = 0
        for notice in lookup.notices:
            kind = normalise_update_type(notice.update_type)
            if kind in self._ignored:
                ignored += 1
                continue
            status = self._meaning.get(kind)
            if status is None:
                unmapped.append(notice)
                relevant.append((IntegrityStatus.UNKNOWN, notice))
            else:
                relevant.append((status, notice))

        if not relevant:
            return IntegrityObservation(
                paper_id=paper_id,
                identifier=identifier,
                doi=lookup.doi,
                status=IntegrityStatus.CURRENT,
                basis=ObservationBasis.REGISTRY_RECORD,
                registry=lookup.registry,
                notices=lookup.notices,
                observation=(
                    f"{lookup.registry} returned the record with no integrity-relevant update "
                    f"({ignored} ignored update(s)); {lookup.detail}"
                ),
                retrieved_at=stamp,
            )

        status = min((item[0] for item in relevant), key=lambda value: self._rank[value])
        decisive = _latest([notice for found, notice in relevant if found is status])
        basis = (
            ObservationBasis.UNMAPPED_UPDATE
            if status is IntegrityStatus.UNKNOWN
            else ObservationBasis.REGISTRY_RECORD
        )
        types = sorted({normalise_update_type(notice.update_type) for _, notice in relevant})
        return IntegrityObservation(
            paper_id=paper_id,
            identifier=identifier,
            doi=lookup.doi,
            status=status,
            basis=basis,
            registry=lookup.registry,
            raw_update_type=decisive.update_type,
            source=decisive.source,
            notice_doi=decisive.notice_doi,
            notice_date=decisive.notice_date,
            notices=lookup.notices,
            observation=(
                f"{lookup.registry} lists update(s) {types}; decisive: {decisive.update_type!r} "
                f"from {decisive.source or 'unnamed source'}"
                f"{f' (notice {decisive.notice_doi})' if decisive.notice_doi else ''}"
                + (f"; unmapped types {[n.update_type for n in unmapped]}" if unmapped else "")
            ),
            retrieved_at=stamp,
        )


def _latest(notices: list[UpdateNotice]) -> UpdateNotice:
    """The most recent notice; undated notices sort first so a dated one wins."""
    return max(notices, key=lambda notice: notice.notice_date or "")
