"""JSON-backed canonical document store.

One file per paper under ``storage/papers/documents/``. Documents are large, so unlike
the paper index they are read on demand rather than listed eagerly — ``list_ids`` walks
filenames and never deserialises.

Writes are atomic: a half-written document would deserialise as a valid but truncated
paper, which is worse than no document at all.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from pydantic import ValidationError

from researchagent.core.exceptions import RepositoryError
from researchagent.core.interfaces.document_repository import DocumentRepository
from researchagent.core.logging import get_logger
from researchagent.models.library import storage_key_for
from researchagent.schemas.validated import ValidatedDocument

logger = get_logger(__name__)


class JsonDocumentRepository(DocumentRepository):
    def __init__(self, documents_dir: Path) -> None:
        self._documents_dir = documents_dir
        self._lock = asyncio.Lock()

    @property
    def documents_dir(self) -> Path:
        return self._documents_dir

    async def get(self, paper_id: str) -> ValidatedDocument | None:
        path = self._path_for(paper_id)
        if not path.is_file():
            return None
        try:
            return ValidatedDocument.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValidationError, json.JSONDecodeError) as exc:
            raise RepositoryError(
                "Could not read stored document",
                paper_id=paper_id,
                file=str(path),
                reason=str(exc),
            ) from exc

    async def save(self, document: ValidatedDocument) -> ValidatedDocument:
        async with self._lock:
            path = self._path_for(document.value.paper_id)
            self._documents_dir.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".json.tmp")
            try:
                temporary.write_text(document.model_dump_json(indent=2), encoding="utf-8")
                temporary.replace(path)
            except OSError as exc:
                temporary.unlink(missing_ok=True)
                raise RepositoryError(
                    "Could not write document",
                    paper_id=document.value.paper_id,
                    file=str(path),
                    reason=str(exc),
                ) from exc
            logger.debug("document_persisted", paper_id=document.value.paper_id, file=str(path))
            return document

    async def exists(self, paper_id: str) -> bool:
        return self._path_for(paper_id).is_file()

    async def list_ids(self) -> list[str]:
        if not self._documents_dir.is_dir():
            return []
        return sorted(path.stem for path in self._documents_dir.glob("*.json"))

    async def delete(self, paper_id: str) -> bool:
        path = self._path_for(paper_id)
        if not path.is_file():
            return False
        path.unlink()
        return True

    def _path_for(self, paper_id: str) -> Path:
        try:
            return self._documents_dir / f"{storage_key_for(paper_id)}.json"
        except ValueError as exc:
            raise RepositoryError("Invalid paper id", paper_id=paper_id) from exc
