"""Library records: what the system knows about a paper *and* how far it has processed it.

The pipeline flags are the plug point for later versions — v0.4 sets ``parsed``, v0.5
sets ``chunked``/``embedded``, v0.6 sets ``verified``. Each stage flips its own flag and
nothing else, so a re-run only does outstanding work.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from researchagent.models.paper import Paper

_UNSAFE_PATH_CHARS = re.compile(r"[^A-Za-z0-9._-]+")

_PIPELINE_FLAGS = (
    "downloaded",
    "parsed",
    "chunked",
    "embedded",
    "extracted",
    "verified",
    "indexed_in_graph",
)


class ProcessingStatus(BaseModel):
    """Pipeline progress for one paper. Every flag is owned by exactly one version."""

    downloaded: bool = False
    parsed: bool = False
    chunked: bool = False
    embedded: bool = False
    extracted: bool = False
    verified: bool = False
    indexed_in_graph: bool = False

    last_error: str | None = None
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def mark(self, **flags: object) -> ProcessingStatus:
        return self.model_copy(update={**flags, "updated_at": datetime.now(UTC)})

    def merge(self, other: ProcessingStatus) -> ProcessingStatus:
        """Union of progress from two views of the same paper.

        Monotonic on purpose: a stage may only ever advance a flag. Re-discovering a
        paper must not reset work already done, and a stage reporting new progress must
        not be overwritten by the stored defaults.
        """
        flags = {
            name: bool(getattr(self, name) or getattr(other, name)) for name in _PIPELINE_FLAGS
        }
        newer = self if self.updated_at >= other.updated_at else other
        return self.model_copy(
            update={**flags, "last_error": newer.last_error, "updated_at": newer.updated_at}
        )


class PaperRecord(BaseModel):
    """One paper as persisted: metadata + pipeline state + where the PDF lives.

    Stored as a JSON sidecar so the original PDFs are never touched or moved.
    """

    paper: Paper
    processing: ProcessingStatus = Field(default_factory=ProcessingStatus)
    pdf_path: Path | None = None
    discovered_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    # Run ids that surfaced this paper; a paper found by several reviews keeps them all.
    run_ids: list[str] = Field(default_factory=list)

    @property
    def id(self) -> str:
        return self.paper.id

    @property
    def storage_key(self) -> str:
        return storage_key_for(self.paper.id)

    def touch(self) -> PaperRecord:
        return self.model_copy(update={"updated_at": datetime.now(UTC)})


def storage_key_for(paper_id: str) -> str:
    """Filesystem-safe form of a paper id, used as the metadata filename.

    ``doi:10.1145/3600006`` -> ``doi-10.1145-3600006``. The namespace prefix is kept so
    ids from different providers can never collide in one directory.
    """
    key = _UNSAFE_PATH_CHARS.sub("-", paper_id).strip("-")
    if not key:
        raise ValueError(f"paper id produces an empty storage key: {paper_id!r}")
    # Keep well clear of filesystem name limits while staying unique in practice.
    return key[:180]
