"""JSON-sidecar paper repository.

One file per paper under ``storage/papers/metadata/<key>.json``. Chosen over a database
for v0.3 because the records are diffable, greppable and survive a wiped container, and
because the PDFs themselves are never touched — the sidecar sits beside the collection
rather than reorganising it.

The port is what downstream code depends on, so swapping this for PostgreSQL later is an
adapter change.

Writes are atomic (temp file + replace) and serialised per process, because v0.4 will
parse papers concurrently and a half-written record would be indistinguishable from a
valid one.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from researchagent.core.exceptions import RepositoryError
from researchagent.core.interfaces.paper_repository import PaperRepository
from researchagent.core.logging import get_logger
from researchagent.models.library import PaperRecord, storage_key_for

logger = get_logger(__name__)


class JsonPaperRepository(PaperRepository):
    def __init__(self, metadata_dir: Path) -> None:
        self._metadata_dir = metadata_dir
        self._lock = asyncio.Lock()

    @property
    def metadata_dir(self) -> Path:
        return self._metadata_dir

    async def get(self, paper_id: str) -> PaperRecord | None:
        path = self._path_for(paper_id)
        if not path.is_file():
            return None
        return self._read(path)

    async def save(self, record: PaperRecord) -> PaperRecord:
        async with self._lock:
            return self._save_unlocked(record)

    async def save_many(self, records: Sequence[PaperRecord]) -> list[PaperRecord]:
        async with self._lock:
            return [self._save_unlocked(record) for record in records]

    async def list_all(self) -> list[PaperRecord]:
        if not self._metadata_dir.is_dir():
            return []
        records = []
        for path in sorted(self._metadata_dir.glob("*.json")):
            try:
                records.append(self._read(path))
            except RepositoryError as exc:
                # One corrupt sidecar must not make the whole library unreadable.
                logger.error("paper_record_unreadable", file=str(path), error=exc.message)
        return records

    async def exists(self, paper_id: str) -> bool:
        return self._path_for(paper_id).is_file()

    async def delete(self, paper_id: str) -> bool:
        path = self._path_for(paper_id)
        if not path.is_file():
            return False
        path.unlink()
        return True

    def _save_unlocked(self, record: PaperRecord) -> PaperRecord:
        path = self._path_for(record.id)
        merged = self._merge_with_existing(record, path)

        self._metadata_dir.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        try:
            temporary.write_text(
                merged.model_dump_json(indent=2, exclude_none=False), encoding="utf-8"
            )
            temporary.replace(path)
        except OSError as exc:
            temporary.unlink(missing_ok=True)
            raise RepositoryError(
                "Could not write paper record", paper_id=record.id, file=str(path), reason=str(exc)
            ) from exc
        return merged

    def _merge_with_existing(self, record: PaperRecord, path: Path) -> PaperRecord:
        """Preserve pipeline progress across re-discovery.

        Processing flags are unioned rather than overwritten in either direction: a
        re-discovered paper must not reset ``parsed``/``embedded``, and a stage reporting
        new progress must not be reverted to the stored state.
        """
        if not path.is_file():
            return record.touch()

        existing = self._read(path)
        return record.model_copy(
            update={
                "processing": record.processing.merge(existing.processing),
                "pdf_path": record.pdf_path or existing.pdf_path,
                "discovered_at": existing.discovered_at,
                "run_ids": sorted({*existing.run_ids, *record.run_ids}),
            }
        ).touch()

    def _read(self, path: Path) -> PaperRecord:
        try:
            return PaperRecord.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValidationError, json.JSONDecodeError) as exc:
            raise RepositoryError(
                "Could not read paper record", file=str(path), reason=str(exc)
            ) from exc

    def _path_for(self, paper_id: str) -> Path:
        try:
            return self._metadata_dir / f"{storage_key_for(paper_id)}.json"
        except ValueError as exc:
            raise RepositoryError("Invalid paper id", paper_id=paper_id) from exc
