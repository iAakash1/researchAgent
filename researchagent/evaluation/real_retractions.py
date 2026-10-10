"""Real-retraction input: a user-supplied list of DOIs with known integrity status.

Nothing here ships data. The repository contains only the schema and an empty template
(``evaluation/gold/real_retractions.template.json``). Entries are supplied by the user,
each naming where its status came from, and are checked against the live registry only
when the user runs ``scripts/check_real_retractions.py``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

from researchagent.core.exceptions import ConfigurationError
from researchagent.models.integrity import IntegrityObservation, IntegrityStatus, ObservationBasis
from researchagent.services.integrity.service import IntegrityService

_DOI = re.compile(r"^10\.\d{4,9}/\S+$", re.IGNORECASE)


class RealRetractionEntry(BaseModel):
    model_config = {"frozen": True}

    doi: str
    expected_status: IntegrityStatus
    # Where the user got the status, e.g. "Retraction Watch record 20946".
    source: str = Field(min_length=1)
    notes: str = ""

    @field_validator("doi")
    @classmethod
    def _is_a_doi(cls, value: str) -> str:
        doi = value.strip().removeprefix("https://doi.org/").lower()
        if not _DOI.match(doi):
            raise ValueError(f"not a DOI: {value!r}")
        return doi


class RealRetractionSet(BaseModel):
    model_config = {"frozen": True}

    version: str = Field(min_length=1)
    description: str = ""
    entries: tuple[RealRetractionEntry, ...] = ()

    @classmethod
    def load(cls, path: Path) -> RealRetractionSet:
        if not path.is_file():
            raise ConfigurationError("Real-retraction file not found", file=str(path))
        try:
            return cls.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ConfigurationError(
                "Real-retraction file could not be read", file=str(path), reason=str(exc)
            ) from exc


class RegistryAgreement(BaseModel):
    model_config = {"frozen": True}

    doi: str
    expected: IntegrityStatus
    observed: IntegrityStatus
    basis: ObservationBasis
    agrees: bool | None = Field(description="None when the lookup failed: no comparison")


class RealRetractionReport(BaseModel):
    model_config = {"frozen": True}

    version: str
    entries: int
    agreed: int
    disagreed: int
    lookup_failures: int
    agreement_rate: float | None
    rows: tuple[RegistryAgreement, ...]


async def check_against_registry(
    dataset: RealRetractionSet, integrity: IntegrityService, *, run_id: str | None = None
) -> RealRetractionReport:
    """Compare each user-supplied status with what the registry says now."""
    refreshed = await integrity.refresh(
        [f"doi:{entry.doi}" for entry in dataset.entries], run_id=run_id
    )
    observed = {item.paper_id: item.effective or item.observed for item in refreshed}
    rows: list[RegistryAgreement] = []
    for entry in dataset.entries:
        found: IntegrityObservation = observed[f"doi:{entry.doi}"]
        rows.append(
            RegistryAgreement(
                doi=entry.doi,
                expected=entry.expected_status,
                observed=found.status,
                basis=found.basis,
                agrees=(found.status is entry.expected_status) if found.is_informative else None,
            )
        )
    agreed = sum(1 for row in rows if row.agrees is True)
    disagreed = sum(1 for row in rows if row.agrees is False)
    compared = agreed + disagreed
    return RealRetractionReport(
        version=dataset.version,
        entries=len(rows),
        agreed=agreed,
        disagreed=disagreed,
        lookup_failures=len(rows) - compared,
        agreement_rate=round(agreed / compared, 6) if compared else None,
        rows=tuple(rows),
    )
