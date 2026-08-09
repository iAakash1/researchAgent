"""JSON-backed bundle store."""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

from pydantic import ValidationError

from researchagent.core.exceptions import RepositoryError
from researchagent.core.interfaces.bundle_repository import BundleRepository
from researchagent.core.logging import get_logger
from researchagent.models.bundle import EvidenceBundle

logger = get_logger(__name__)

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


class JsonBundleRepository(BundleRepository):
    def __init__(self, bundles_dir: Path) -> None:
        self._bundles_dir = bundles_dir
        self._lock = asyncio.Lock()

    @property
    def bundles_dir(self) -> Path:
        return self._bundles_dir

    async def get(self, bundle_id: str) -> EvidenceBundle | None:
        path = self._path_for(bundle_id)
        if not path.is_file():
            return None
        return self._read(path)

    async def save(self, bundle: EvidenceBundle) -> EvidenceBundle:
        async with self._lock:
            path = self._path_for(bundle.id)
            self._bundles_dir.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".json.tmp")
            try:
                temporary.write_text(bundle.model_dump_json(indent=2), encoding="utf-8")
                temporary.replace(path)
            except OSError as exc:
                temporary.unlink(missing_ok=True)
                raise RepositoryError(
                    "Could not write bundle", bundle_id=bundle.id, reason=str(exc)
                ) from exc
            logger.debug(
                "bundle_persisted",
                bundle_id=bundle.id,
                objects=len(bundle.knowledge_objects),
                evidence=len(bundle.evidence),
            )
            return bundle

    async def for_question(self, question_id: str) -> tuple[EvidenceBundle, ...]:
        return tuple(
            bundle for bundle in await self._all() if bundle.query.question_id == question_id
        )

    async def list_ids(self) -> list[str]:
        if not self._bundles_dir.is_dir():
            return []
        return sorted(path.stem for path in self._bundles_dir.glob("*.json"))

    async def delete(self, bundle_id: str) -> bool:
        path = self._path_for(bundle_id)
        if not path.is_file():
            return False
        path.unlink()
        return True

    async def _all(self) -> list[EvidenceBundle]:
        if not self._bundles_dir.is_dir():
            return []
        bundles = []
        for path in sorted(self._bundles_dir.glob("*.json")):
            try:
                bundles.append(self._read(path))
            except RepositoryError as exc:
                logger.error("bundle_unreadable", file=str(path), error=exc.message)
        return bundles

    def _read(self, path: Path) -> EvidenceBundle:
        try:
            return EvidenceBundle.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValidationError, json.JSONDecodeError) as exc:
            raise RepositoryError("Could not read bundle", file=str(path), reason=str(exc)) from exc

    def _path_for(self, bundle_id: str) -> Path:
        key = _UNSAFE.sub("-", bundle_id).strip("-")[:180]
        if not key:
            raise RepositoryError("Invalid bundle id", bundle_id=bundle_id)
        return self._bundles_dir / f"{key}.json"
