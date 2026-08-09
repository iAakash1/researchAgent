"""The evidence index — evidence as an independently addressable citizen.

``core/evidence.py`` defines the primitive: a claim, a quote, a location. This module
defines how evidence is *stored and found* on its own terms.

The separation matters and is the point of the release:

* A :class:`KnowledgeObject` still owns the evidence that admitted it. That ownership is
  what makes "no evidence, no knowledge" enforceable at construction, and it is not
  negotiable — an id-only reference could dangle, which is precisely the hole the
  invariant exists to close.
* Evidence is *additionally* indexed here, keyed by its own id, retrievable without
  going through knowledge at all. Knowledge is not the access path to evidence.
* The relationship between the two is an explicit :class:`EvidenceLink` record. Evidence
  never points at knowledge; the link does, and it can be queried from either side.

So knowledge does not mediate evidence retrieval, evidence does not know about knowledge,
and the association is a first-class, inspectable object.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field

from researchagent.core.evidence import Evidence, EvidenceKind, SourceLocation
from researchagent.models.knowledge import KnowledgeKind


class EvidenceRole(StrEnum):
    """What an evidence item does for the thing it is linked to."""

    # The quote that admitted the object in the first place.
    FOUNDING = "founding"
    # Additional corroboration found later, or merged from a duplicate extraction.
    CORROBORATING = "corroborating"
    # Evidence that argues against the linked object.
    CONTRADICTING = "contradicting"


class EvidenceLink(BaseModel):
    """An explicit association between one evidence item and one knowledge object.

    A separate record rather than a field on either side, so the association can be
    added, revised or contradicted without rewriting the things it connects — and so a
    later stage can attach contradicting evidence to an object without mutating it.
    """

    model_config = {"frozen": True}

    evidence_id: str = Field(min_length=1)
    knowledge_object_id: str = Field(min_length=1)
    knowledge_kind: KnowledgeKind
    role: EvidenceRole = EvidenceRole.FOUNDING
    linked_by: str = Field(min_length=1)
    linked_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class EvidenceRecord(BaseModel):
    """One evidence item as stored: the primitive, plus where it belongs and what it serves."""

    model_config = {"frozen": True}

    evidence: Evidence
    paper_id: str = Field(min_length=1)
    # Ties the evidence to the exact document bytes it was read from. Without this a
    # re-parsed paper would silently invalidate every location while looking unchanged.
    document_sha256: str = Field(min_length=1)
    links: tuple[EvidenceLink, ...] = ()

    @property
    def id(self) -> str:
        return self.evidence.id

    @property
    def kind(self) -> EvidenceKind:
        return self.evidence.kind

    @property
    def location(self) -> SourceLocation:
        return self.evidence.location

    @property
    def quote(self) -> str:
        return self.evidence.quote or ""

    @property
    def knowledge_object_ids(self) -> tuple[str, ...]:
        return tuple(link.knowledge_object_id for link in self.links)

    @property
    def content_hash(self) -> str:
        """Identity by content, not by generated id.

        Two extractors quoting the same sentence for the same paragraph produce the same
        evidence; deduplicating on this rather than on ``evidence.id`` is what stops one
        fact from being counted twice when it was observed twice.
        """
        return content_hash_for(self.paper_id, self.evidence)

    def linked_to(self, link: EvidenceLink) -> EvidenceRecord:
        """Return a copy carrying an additional association. Records are never mutated."""
        if any(existing == link for existing in self.links):
            return self
        return self.model_copy(update={"links": (*self.links, link)})


class PaperEvidence(BaseModel):
    """Every evidence item indexed for one paper.

    Stored per paper because that is the unit that is re-derived together: re-parsing a
    document invalidates all of its evidence at once, and nothing else's.
    """

    model_config = {"frozen": True}

    paper_id: str = Field(min_length=1)
    document_sha256: str = Field(min_length=1)
    records: tuple[EvidenceRecord, ...] = ()
    indexed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def by_id(self, evidence_id: str) -> EvidenceRecord | None:
        return next((record for record in self.records if record.id == evidence_id), None)

    def for_knowledge_object(self, object_id: str) -> tuple[EvidenceRecord, ...]:
        return tuple(record for record in self.records if object_id in record.knowledge_object_ids)

    @property
    def sections_covered(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    record.location.section_title
                    for record in self.records
                    if record.location.section_title
                }
            )
        )


def content_hash_for(paper_id: str, evidence: Evidence) -> str:
    """Stable identity for an evidence item: paper, location and quoted text."""
    location = evidence.location
    parts = (
        paper_id,
        str(location.section_id),
        str(location.paragraph_index),
        " ".join((evidence.quote or evidence.claim).split()).lower(),
    )
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:20]  # noqa: S324
