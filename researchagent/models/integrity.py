"""Post-publication integrity of a paper: retractions, withdrawals, concerns, corrections.

A language model cannot be asked whether a paper has been retracted: offline models miss
most retractions, and an answer it cannot cite is an opinion. Integrity status here is
only ever read off a registry record, and it exists only as an
:class:`IntegrityObservation` — a status with the observation that produced it, exactly as
a ``ConfidenceSignal`` cannot exist without its ``observation``.

The statuses are kept distinct on purpose. A correction or an expression of concern is not
a retraction, and collapsing them (as OpenAlex's ``is_retracted`` does) would invalidate
conclusions whose sources are merely under discussion.

A lookup that failed, or that could not be resolved, is ``UNKNOWN`` — never ``CURRENT``.
"Nobody could check" and "checked and found clean" must stay distinguishable.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator


class IntegrityStatus(StrEnum):
    """What the scholarly record currently says about a paper."""

    CURRENT = "current"
    RETRACTED = "retracted"
    WITHDRAWN = "withdrawn"
    EXPRESSION_OF_CONCERN = "expression_of_concern"
    CORRECTED = "corrected"
    UNKNOWN = "unknown"


class ObservationBasis(StrEnum):
    """How an observation's status was arrived at."""

    # The registry returned the work; the status is derived from its update metadata.
    REGISTRY_RECORD = "registry_record"
    # The registry reported an update whose type has no configured meaning.
    UNMAPPED_UPDATE = "unmapped_update"
    # The paper carries no identifier this registry can resolve.
    NO_LOOKUP_IDENTIFIER = "no_lookup_identifier"
    # Network, HTTP or parse failure, or the registry has no record.
    LOOKUP_FAILED = "lookup_failed"
    # Secondary signal: the parsed PDF text contains a retraction or concern notice.
    PDF_NOTICE = "pdf_notice"
    # Set by the evaluation harness when it flips a status deliberately.
    INJECTED = "injected"

    @property
    def is_informative(self) -> bool:
        """Whether this observation says anything about the paper at all.

        A failed lookup is a fact about the lookup, not about the paper, so it never
        replaces an earlier informative observation.
        """
        return self not in (ObservationBasis.NO_LOOKUP_IDENTIFIER, ObservationBasis.LOOKUP_FAILED)

    @property
    def flags_review(self) -> bool:
        """An ``UNKNOWN`` that still asks a human to look."""
        return self in (ObservationBasis.UNMAPPED_UPDATE, ObservationBasis.PDF_NOTICE)


_REGISTRY_STATUSES = frozenset(
    {
        IntegrityStatus.RETRACTED,
        IntegrityStatus.WITHDRAWN,
        IntegrityStatus.EXPRESSION_OF_CONCERN,
        IntegrityStatus.CORRECTED,
    }
)


class UpdateNotice(BaseModel):
    """One editorial update as a registry reported it, kept verbatim."""

    model_config = {"frozen": True}

    notice_doi: str | None = None
    # Raw as returned, e.g. "retraction", "expression_of_concern".
    update_type: str = Field(min_length=1)
    # Who asserted it, as returned ("publisher", "retraction-watch").
    source: str | None = None
    label: str | None = None
    # ISO partial date ("2019", "2019-07", "2019-07-26"). Never padded: a year-only
    # notice must not acquire an invented month.
    notice_date: str | None = None
    record_id: str | None = None
    # Which side of the relation it was read from: the work's `updated-by`, or a notice's
    # `update-to` pointing at the work.
    relation: str = Field(min_length=1)


class IntegrityObservation(BaseModel):
    """A paper's integrity status and the observation it was derived from."""

    model_config = {"frozen": True}

    id: str = Field(default_factory=lambda: f"IO-{uuid4().hex[:12]}")
    paper_id: str = Field(min_length=1)
    # The identifier the lookup used or attempted, e.g. "doi:10.1021/am300292v".
    identifier: str = Field(min_length=1)
    doi: str | None = None
    status: IntegrityStatus
    basis: ObservationBasis
    registry: str = Field(min_length=1)
    # The decisive notice, when there is one.
    raw_update_type: str | None = None
    source: str | None = None
    notice_doi: str | None = None
    notice_date: str | None = None
    notices: tuple[UpdateNotice, ...] = ()
    observation: str = Field(min_length=1, description="The checkable fact behind `status`")
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="after")
    def _status_agrees_with_basis(self) -> IntegrityObservation:
        if (
            self.status is IntegrityStatus.CURRENT
            and self.basis is not ObservationBasis.REGISTRY_RECORD
        ):
            raise ValueError("CURRENT can only come from a registry record, never from a failure")
        if (
            self.basis
            in (
                ObservationBasis.UNMAPPED_UPDATE,
                ObservationBasis.NO_LOOKUP_IDENTIFIER,
                ObservationBasis.LOOKUP_FAILED,
                ObservationBasis.PDF_NOTICE,
            )
            and self.status is not IntegrityStatus.UNKNOWN
        ):
            raise ValueError(f"a {self.basis.value} observation can only be UNKNOWN")
        if self.status in _REGISTRY_STATUSES:
            if self.basis not in (ObservationBasis.REGISTRY_RECORD, ObservationBasis.INJECTED):
                raise ValueError(f"{self.status.value} requires a registry record")
            if self.basis is ObservationBasis.REGISTRY_RECORD and not self.raw_update_type:
                raise ValueError(f"{self.status.value} must name its registry update type")
        if self.basis is ObservationBasis.INJECTED and self.status is IntegrityStatus.CURRENT:
            raise ValueError("an injected observation cannot assert CURRENT")
        return self

    @property
    def is_informative(self) -> bool:
        return self.basis.is_informative

    @property
    def flags_review(self) -> bool:
        return self.status is IntegrityStatus.UNKNOWN and self.basis.flags_review


class IntegrityChange(BaseModel):
    """One paper's integrity, before and after a refresh.

    ``before`` is the latest *informative* observation held before the refresh, or None
    when the paper had never been checked — which is not the same as having been CURRENT.
    """

    model_config = {"frozen": True}

    paper_id: str = Field(min_length=1)
    before: IntegrityObservation | None = None
    after: IntegrityObservation
    # Whether `after` came from the cache rather than a new lookup.
    cached: bool = False

    @property
    def is_material(self) -> bool:
        """Whether the effective status moved.

        An uninformative lookup moves nothing. A first observation is material only when
        it reports something other than CURRENT: a conclusion was accepted on the
        assumption that its sources stood, and confirming that is not news.
        """
        if not self.after.is_informative:
            return False
        if self.before is None:
            return self.after.status is not IntegrityStatus.CURRENT
        return self.before.status is not self.after.status
