"""Integrity registry port.

A registry answers one question: which editorial updates (retraction, withdrawal,
expression of concern, correction, ...) has the scholarly record attached to this work?
It reports what it was told, verbatim. Deciding what those updates *mean* for a paper is
configuration-driven domain logic in ``services/integrity/``, not the adapter's job.

Adapters never raise for an expected failure. A timeout or an unknown DOI is a
``LookupOutcome``, because "the check could not be done" is itself something the caller
must record rather than something it may skip.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import UTC, datetime
from enum import StrEnum
from typing import ClassVar

from pydantic import BaseModel, Field

from researchagent.models.integrity import UpdateNotice


class LookupOutcome(StrEnum):
    FOUND = "found"
    # The registry answered and has no record of this identifier.
    NOT_FOUND = "not_found"
    # Network, HTTP, rate-limit or payload failure.
    FAILED = "failed"


class RegistryLookup(BaseModel):
    """What one registry said about one DOI."""

    model_config = {"frozen": True}

    registry: str = Field(min_length=1)
    doi: str = Field(min_length=1)
    outcome: LookupOutcome
    notices: tuple[UpdateNotice, ...] = ()
    detail: str = Field(min_length=1, description="What happened, for the observation")
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class IntegrityRegistry(ABC):
    """A deterministic source of post-publication update metadata."""

    name: ClassVar[str]

    @abstractmethod
    async def lookup(self, doi: str) -> RegistryLookup:
        """Every update the registry links to ``doi``. Must not raise for a failed lookup."""

    @abstractmethod
    async def aclose(self) -> None: ...
