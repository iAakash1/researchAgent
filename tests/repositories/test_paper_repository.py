from __future__ import annotations

import json
from pathlib import Path

import pytest

from researchagent.core.exceptions import RepositoryError
from researchagent.models.library import PaperRecord, storage_key_for
from researchagent.models.paper import Paper, PaperIdentifiers, SourceName
from researchagent.repositories.paper_repository import JsonPaperRepository


def record(paper_id: str = "doi:10.1145/123", **paper_overrides: object) -> PaperRecord:
    return PaperRecord(
        paper=Paper.model_validate(
            {
                "id": paper_id,
                "title": "Metastable failures in distributed systems",
                "provider": SourceName.ARXIV,
            }
            | paper_overrides
        )
    )


@pytest.fixture
def repository(tmp_path: Path) -> JsonPaperRepository:
    return JsonPaperRepository(tmp_path / "metadata")


async def test_save_and_get_round_trip(repository: JsonPaperRepository) -> None:
    saved = await repository.save(record())

    loaded = await repository.get(saved.id)

    assert loaded is not None
    assert loaded.paper.title == "Metastable failures in distributed systems"
    assert loaded.paper.provider is SourceName.ARXIV


async def test_metadata_is_a_readable_json_sidecar(repository: JsonPaperRepository) -> None:
    """The whole point of JSON files: a human can read one without the application."""
    await repository.save(record("manual:01"))

    path = repository.metadata_dir / "manual-01.json"
    payload = json.loads(path.read_text())

    assert payload["paper"]["id"] == "manual:01"
    assert payload["processing"]["parsed"] is False
    assert "discovered_at" in payload


async def test_ids_are_namespaced_so_sources_cannot_collide() -> None:
    assert storage_key_for("manual:01") == "manual-01"
    assert storage_key_for("arxiv:2401.12345") == "arxiv-2401.12345"
    assert storage_key_for("doi:10.1145/3600006") == "doi-10.1145-3600006"
    assert storage_key_for("manual:01") != storage_key_for("arxiv:01")


async def test_missing_record_is_none(repository: JsonPaperRepository) -> None:
    assert await repository.get("doi:nope") is None
    assert await repository.exists("doi:nope") is False


async def test_resaving_preserves_pipeline_progress(
    repository: JsonPaperRepository, tmp_path: Path
) -> None:
    """A second discovery run must never reset work done by a later stage."""
    first = await repository.save(record())
    await repository.save(
        first.model_copy(
            update={
                "processing": first.processing.mark(downloaded=True, parsed=True),
                "pdf_path": tmp_path / "a.pdf",
            }
        )
    )

    # Re-discovery produces a fresh record with default flags.
    await repository.save(record())

    stored = await repository.get(record().id)
    assert stored is not None
    assert stored.processing.parsed is True
    assert stored.processing.downloaded is True
    assert stored.pdf_path == tmp_path / "a.pdf"


async def test_resaving_accumulates_run_ids(repository: JsonPaperRepository) -> None:
    await repository.save(record().model_copy(update={"run_ids": ["run-1"]}))
    await repository.save(record().model_copy(update={"run_ids": ["run-2"]}))

    stored = await repository.get(record().id)
    assert stored is not None
    assert stored.run_ids == ["run-1", "run-2"]


async def test_resaving_refreshes_metadata(repository: JsonPaperRepository) -> None:
    await repository.save(record())
    await repository.save(record(identifiers=PaperIdentifiers(doi="10.1145/123"), year=2024))

    stored = await repository.get(record().id)
    assert stored is not None
    assert stored.paper.year == 2024


async def test_save_many_and_list_all(repository: JsonPaperRepository) -> None:
    await repository.save_many([record("manual:01"), record("manual:02"), record("arxiv:9")])

    assert len(await repository.list_all()) == 3


async def test_list_all_on_empty_repository(repository: JsonPaperRepository) -> None:
    assert await repository.list_all() == []


async def test_delete(repository: JsonPaperRepository) -> None:
    await repository.save(record())

    assert await repository.delete(record().id) is True
    assert await repository.delete(record().id) is False
    assert await repository.get(record().id) is None


async def test_find_pending_is_the_hook_for_later_stages(
    repository: JsonPaperRepository,
) -> None:
    done = record("manual:01")
    await repository.save(done.model_copy(update={"processing": done.processing.mark(parsed=True)}))
    await repository.save(record("manual:02"))

    pending = await repository.find_pending("parsed")

    assert [item.id for item in pending] == ["manual:02"]


async def test_one_corrupt_file_does_not_break_the_library(
    repository: JsonPaperRepository,
) -> None:
    await repository.save(record("manual:01"))
    (repository.metadata_dir / "broken.json").write_text("{not json")

    records = await repository.list_all()

    assert [item.id for item in records] == ["manual:01"]


async def test_reading_a_corrupt_record_directly_raises(
    repository: JsonPaperRepository,
) -> None:
    await repository.save(record("manual:01"))
    (repository.metadata_dir / "manual-01.json").write_text("{not json")

    with pytest.raises(RepositoryError):
        await repository.get("manual:01")


async def test_writes_are_atomic_leaving_no_partial_files(
    repository: JsonPaperRepository,
) -> None:
    await repository.save(record())

    assert list(repository.metadata_dir.glob("*.tmp")) == []
