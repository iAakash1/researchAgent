"""JSON-backed integrity observation cache: one file per paper, appended in order."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from pydantic import TypeAdapter, ValidationError

from researchagent.core.exceptions import RepositoryError
from researchagent.core.interfaces.repositories import IntegrityObservationRepository
from researchagent.core.logging import get_logger
from researchagent.models.integrity import IntegrityObservation
from researchagent.models.library import storage_key_for

logger = get_logger(__name__)

_HISTORY = TypeAdapter(tuple[IntegrityObservation, ...])


class JsonIntegrityRepository(IntegrityObservationRepository):
    def __init__(self, observations_dir: Path) -> None:
        self._dir = observations_dir
        self._lock = asyncio.Lock()

    async def append(self, observation: IntegrityObservation) -> IntegrityObservation:
        async with self._lock:
            history = (*self._read(observation.paper_id), observation)
            path = self._path_for(observation.paper_id)
            self._dir.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".json.tmp")
            try:
                temporary.write_bytes(_HISTORY.dump_json(history, indent=2))
                temporary.replace(path)
            except OSError as exc:
                temporary.unlink(missing_ok=True)
                raise RepositoryError(
                    "Could not write integrity observations",
                    paper_id=observation.paper_id,
                    reason=str(exc),
                ) from exc
        logger.debug(
            "integrity_observation_persisted",
            paper_id=observation.paper_id,
            status=observation.status.value,
            basis=observation.basis.value,
        )
        return observation

    async def history(self, paper_id: str) -> tuple[IntegrityObservation, ...]:
        return self._read(paper_id)

    def _read(self, paper_id: str) -> tuple[IntegrityObservation, ...]:
        path = self._path_for(paper_id)
        if not path.is_file():
            return ()
        try:
            return _HISTORY.validate_json(path.read_bytes())
        except (OSError, ValidationError, json.JSONDecodeError) as exc:
            raise RepositoryError(
                "Could not read integrity observations", paper_id=paper_id, reason=str(exc)
            ) from exc

    def _path_for(self, paper_id: str) -> Path:
        try:
            return self._dir / f"{storage_key_for(paper_id)}.json"
        except ValueError as exc:
            raise RepositoryError("Invalid paper id", paper_id=paper_id) from exc
