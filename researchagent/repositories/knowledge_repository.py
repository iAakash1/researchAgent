"""JSON-backed knowledge store.

One file per paper under ``storage/papers/knowledge/``. Atomic writes, for the same
reason as the document store: a truncated file deserialises into a valid-looking object
with facts missing, which is the worst possible failure for a knowledge base.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from pydantic import ValidationError

from researchagent.core.exceptions import RepositoryError
from researchagent.core.interfaces.knowledge_repository import KnowledgeRepository
from researchagent.core.logging import get_logger
from researchagent.models.library import storage_key_for
from researchagent.schemas.knowledge import ValidatedKnowledge

logger = get_logger(__name__)


class JsonKnowledgeRepository(KnowledgeRepository):
    def __init__(self, knowledge_dir: Path) -> None:
        self._knowledge_dir = knowledge_dir
        self._lock = asyncio.Lock()

    @property
    def knowledge_dir(self) -> Path:
        return self._knowledge_dir

    async def get(self, paper_id: str) -> ValidatedKnowledge | None:
        path = self._path_for(paper_id)
        if not path.is_file():
            return None
        try:
            return ValidatedKnowledge.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValidationError, json.JSONDecodeError) as exc:
            raise RepositoryError(
                "Could not read stored knowledge",
                paper_id=paper_id,
                file=str(path),
                reason=str(exc),
            ) from exc

    async def save(self, knowledge: ValidatedKnowledge) -> ValidatedKnowledge:
        async with self._lock:
            path = self._path_for(knowledge.value.paper_id)
            self._knowledge_dir.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".json.tmp")
            try:
                temporary.write_text(knowledge.model_dump_json(indent=2), encoding="utf-8")
                temporary.replace(path)
            except OSError as exc:
                temporary.unlink(missing_ok=True)
                raise RepositoryError(
                    "Could not write knowledge",
                    paper_id=knowledge.value.paper_id,
                    file=str(path),
                    reason=str(exc),
                ) from exc
            logger.debug(
                "knowledge_persisted",
                paper_id=knowledge.value.paper_id,
                objects=len(knowledge.value.objects),
            )
            return knowledge

    async def exists(self, paper_id: str) -> bool:
        return self._path_for(paper_id).is_file()

    async def list_ids(self) -> list[str]:
        if not self._knowledge_dir.is_dir():
            return []
        return sorted(path.stem for path in self._knowledge_dir.glob("*.json"))

    async def delete(self, paper_id: str) -> bool:
        path = self._path_for(paper_id)
        if not path.is_file():
            return False
        path.unlink()
        return True

    def _path_for(self, paper_id: str) -> Path:
        try:
            return self._knowledge_dir / f"{storage_key_for(paper_id)}.json"
        except ValueError as exc:
            raise RepositoryError("Invalid paper id", paper_id=paper_id) from exc
