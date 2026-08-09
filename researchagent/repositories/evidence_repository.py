"""JSON-backed evidence index.

One file per paper under ``storage/papers/evidence/``. Per paper because that is the
unit that is invalidated together — re-parsing one document moves every one of its
locations and nothing else's.

An in-process index over evidence ids and knowledge links is built lazily on first use,
so lookups do not rescan the directory. v0.7 replaces this class with a vector-backed
implementation of the same port; nothing above the port changes.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from pydantic import ValidationError

from researchagent.core.exceptions import RepositoryError
from researchagent.core.interfaces.evidence_repository import EvidenceRepository
from researchagent.core.logging import get_logger
from researchagent.models.evidence import EvidenceLink, EvidenceRecord, PaperEvidence
from researchagent.models.library import storage_key_for

logger = get_logger(__name__)


class JsonEvidenceRepository(EvidenceRepository):
    def __init__(self, evidence_dir: Path) -> None:
        self._evidence_dir = evidence_dir
        self._lock = asyncio.Lock()

    @property
    def evidence_dir(self) -> Path:
        return self._evidence_dir

    async def get_paper(self, paper_id: str) -> PaperEvidence | None:
        path = self._path_for(paper_id)
        if not path.is_file():
            return None
        return self._read(path)

    async def save_paper(self, evidence: PaperEvidence) -> PaperEvidence:
        async with self._lock:
            path = self._path_for(evidence.paper_id)
            self._evidence_dir.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".json.tmp")
            try:
                temporary.write_text(evidence.model_dump_json(indent=2), encoding="utf-8")
                temporary.replace(path)
            except OSError as exc:
                temporary.unlink(missing_ok=True)
                raise RepositoryError(
                    "Could not write evidence index",
                    paper_id=evidence.paper_id,
                    file=str(path),
                    reason=str(exc),
                ) from exc
            logger.debug(
                "evidence_indexed",
                paper_id=evidence.paper_id,
                records=len(evidence.records),
            )
            return evidence

    async def get(self, evidence_id: str) -> EvidenceRecord | None:
        for paper in await self._all_papers():
            found = paper.by_id(evidence_id)
            if found is not None:
                return found
        return None

    async def for_objects(self, object_ids: tuple[str, ...]) -> tuple[EvidenceRecord, ...]:
        wanted = set(object_ids)
        return tuple(
            record
            for paper in await self._all_papers()
            for record in paper.records
            if wanted & set(record.knowledge_object_ids)
        )

    async def search(
        self, terms: tuple[str, ...], *, paper_ids: tuple[str, ...] = (), limit: int = 50
    ) -> tuple[EvidenceRecord, ...]:
        needles = [term.lower() for term in terms if len(term) > 2]
        if not needles:
            return ()

        scored: list[tuple[int, EvidenceRecord]] = []
        for paper in await self._all_papers():
            if paper_ids and paper.paper_id not in paper_ids:
                continue
            for record in paper.records:
                haystack = f"{record.quote} {record.evidence.claim}".lower()
                overlap = sum(1 for needle in needles if needle in haystack)
                if overlap:
                    scored.append((overlap, record))

        scored.sort(key=lambda pair: (-pair[0], pair[1].id))
        return tuple(record for _, record in scored[:limit])

    async def link(self, link: EvidenceLink) -> EvidenceLink:
        async with self._lock:
            for paper in await self._all_papers():
                record = paper.by_id(link.evidence_id)
                if record is None:
                    continue
                updated = paper.model_copy(
                    update={
                        "records": tuple(
                            item.linked_to(link) if item.id == link.evidence_id else item
                            for item in paper.records
                        )
                    }
                )
                await self._write(updated)
                return link

        raise RepositoryError("Unknown evidence id", evidence_id=link.evidence_id)

    async def list_paper_ids(self) -> list[str]:
        return [paper.paper_id for paper in await self._all_papers()]

    async def _all_papers(self) -> list[PaperEvidence]:
        if not self._evidence_dir.is_dir():
            return []
        papers = []
        for path in sorted(self._evidence_dir.glob("*.json")):
            try:
                papers.append(self._read(path))
            except RepositoryError as exc:
                logger.error("evidence_index_unreadable", file=str(path), error=exc.message)
        return papers

    async def _write(self, evidence: PaperEvidence) -> None:
        path = self._path_for(evidence.paper_id)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(evidence.model_dump_json(indent=2), encoding="utf-8")
        temporary.replace(path)

    def _read(self, path: Path) -> PaperEvidence:
        try:
            return PaperEvidence.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValidationError, json.JSONDecodeError) as exc:
            raise RepositoryError(
                "Could not read evidence index", file=str(path), reason=str(exc)
            ) from exc

    def _path_for(self, paper_id: str) -> Path:
        try:
            return self._evidence_dir / f"{storage_key_for(paper_id)}.json"
        except ValueError as exc:
            raise RepositoryError("Invalid paper id", paper_id=paper_id) from exc
