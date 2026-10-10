"""JSON-backed persistence for conclusion maintenance.

Layout (all under ``config/maintenance.yaml`` ``storage``)::

    conclusions/<conclusion>/v0001.json, v0002.json, ...   append-only, never rewritten
    topics/<topic>.json                                     replaced as the topic advances
    diffs/<topic>/<started_at>-<diff>.json                  one per maintenance run

Version files are created with exclusive-create, so even a concurrent writer cannot
overwrite a version that already exists.
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

from pydantic import BaseModel, ValidationError

from researchagent.core.exceptions import RepositoryError
from researchagent.core.interfaces.repositories import (
    ConclusionRepository,
    MaintenanceDiffRepository,
)
from researchagent.core.logging import get_logger
from researchagent.models.maintenance import ConclusionDiff, ConclusionRecord, MaintainedTopic

logger = get_logger(__name__)

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _key(identifier: str) -> str:
    key = _UNSAFE.sub("-", identifier).strip("-")[:180]
    if not key:
        raise RepositoryError("Identifier produces an empty storage key", identifier=identifier)
    return key


def _read[T: BaseModel](path: Path, schema: type[T]) -> T:
    try:
        return schema.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError, json.JSONDecodeError) as exc:
        raise RepositoryError(
            "Could not read maintenance record", file=str(path), reason=str(exc)
        ) from exc


def _replace(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(path)
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        raise RepositoryError(
            "Could not write maintenance record", file=str(path), reason=str(exc)
        ) from exc


class JsonConclusionRepository(ConclusionRepository):
    def __init__(self, root: Path) -> None:
        self._conclusions = root / "conclusions"
        self._topics = root / "topics"
        self._lock = asyncio.Lock()

    async def save_topic(self, topic: MaintainedTopic) -> MaintainedTopic:
        async with self._lock:
            _replace(self._topics / f"{_key(topic.topic_id)}.json", topic.model_dump_json(indent=2))
        return topic

    async def get_topic(self, topic_id: str) -> MaintainedTopic | None:
        path = self._topics / f"{_key(topic_id)}.json"
        return _read(path, MaintainedTopic) if path.is_file() else None

    async def list_topics(self) -> tuple[MaintainedTopic, ...]:
        if not self._topics.is_dir():
            return ()
        return tuple(_read(path, MaintainedTopic) for path in sorted(self._topics.glob("*.json")))

    async def append(self, record: ConclusionRecord) -> ConclusionRecord:
        async with self._lock:
            history = self._history(record.conclusion_id)
            expected = len(history) + 1
            if record.version != expected:
                raise RepositoryError(
                    "Conclusion history is append-only; only the next version may be written",
                    conclusion_id=record.conclusion_id,
                    version=record.version,
                    expected=expected,
                )
            if history and history[-1].topic_id != record.topic_id:
                raise RepositoryError(
                    "A conclusion cannot move between topics", conclusion_id=record.conclusion_id
                )
            directory = self._conclusions / _key(record.conclusion_id)
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"v{record.version:04d}.json"
            try:
                with path.open("x", encoding="utf-8") as handle:
                    handle.write(record.model_dump_json(indent=2))
            except FileExistsError as exc:
                raise RepositoryError(
                    "Conclusion version already exists",
                    conclusion_id=record.conclusion_id,
                    version=record.version,
                ) from exc
            except OSError as exc:
                path.unlink(missing_ok=True)
                raise RepositoryError(
                    "Could not write conclusion version", file=str(path), reason=str(exc)
                ) from exc
        logger.debug(
            "conclusion_version_persisted",
            conclusion_id=record.conclusion_id,
            version=record.version,
            state=record.state.value,
        )
        return record

    async def history(self, conclusion_id: str) -> tuple[ConclusionRecord, ...]:
        return self._history(conclusion_id)

    async def for_topic(self, topic_id: str) -> tuple[ConclusionRecord, ...]:
        if not self._conclusions.is_dir():
            return ()
        latest: list[ConclusionRecord] = []
        for directory in sorted(path for path in self._conclusions.iterdir() if path.is_dir()):
            versions = sorted(directory.glob("v*.json"))
            if not versions:
                continue
            record = _read(versions[-1], ConclusionRecord)
            if record.topic_id == topic_id:
                latest.append(record)
        return tuple(sorted(latest, key=lambda item: item.conclusion_id))

    def _history(self, conclusion_id: str) -> tuple[ConclusionRecord, ...]:
        directory = self._conclusions / _key(conclusion_id)
        if not directory.is_dir():
            return ()
        return tuple(_read(path, ConclusionRecord) for path in sorted(directory.glob("v*.json")))


class JsonMaintenanceDiffRepository(MaintenanceDiffRepository):
    def __init__(self, root: Path) -> None:
        self._root = root / "diffs"
        self._lock = asyncio.Lock()

    async def save(self, diff: ConclusionDiff) -> ConclusionDiff:
        stamp = diff.started_at.strftime("%Y%m%dT%H%M%S%f")
        path = self._root / _key(diff.topic_id) / f"{stamp}-{_key(diff.id)}.json"
        async with self._lock:
            _replace(path, diff.model_dump_json(indent=2))
        return diff

    async def get(self, diff_id: str) -> ConclusionDiff | None:
        if not self._root.is_dir():
            return None
        match = next(iter(sorted(self._root.glob(f"*/*-{_key(diff_id)}.json"))), None)
        return _read(match, ConclusionDiff) if match is not None else None

    async def for_topic(self, topic_id: str) -> tuple[ConclusionDiff, ...]:
        directory = self._root / _key(topic_id)
        if not directory.is_dir():
            return ()
        return tuple(_read(path, ConclusionDiff) for path in sorted(directory.glob("*.json")))
