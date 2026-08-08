from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from httpx import ASGITransport, AsyncClient

from researchagent.api.app import create_app
from researchagent.container import Container
from researchagent.models.library import PaperRecord
from researchagent.models.paper import Paper, SourceName
from researchagent.repositories.paper_repository import JsonPaperRepository
from tests.agents.test_planner import framing, strategy
from tests.conftest import FakeLLMProvider


@pytest.fixture
def fake_provider() -> FakeLLMProvider:
    """Drives the real Planner: one framing + one strategy reply per request."""
    return FakeLLMProvider(
        structured_sequence=[
            framing(
                topic="Metastable failures in distributed systems",
                questions=[
                    {
                        "question": "What triggers metastable failures in distributed systems?",
                        "rationale": "Triggers determine which mitigations apply at all.",
                        "priority": "high",
                        "keywords": ["metastable", "distributed systems"],
                    }
                ],
            ),
            strategy(queries=["metastable failures distributed systems"]),
        ]
    )


@pytest.fixture
async def client(container: Container) -> AsyncIterator[AsyncClient]:
    app = create_app(container=container)
    transport = ASGITransport(app=app)
    async with (
        AsyncClient(transport=transport, base_url="http://test", timeout=30) as http_client,
        app.router.lifespan_context(app),
    ):
        yield http_client


@pytest.fixture
async def seeded(paper_repository: JsonPaperRepository, manual_source) -> list[PaperRecord]:
    records = [
        PaperRecord(paper=paper, pdf_path=paper.local_path)
        for paper in manual_source.load_all()[:3]
    ]
    records.append(
        PaperRecord(
            paper=Paper(
                id="arxiv:2401.1",
                title="A remote preprint",
                provider=SourceName.ARXIV,
                pdf_url="https://arxiv.org/pdf/2401.1",
            )
        )
    )
    return await paper_repository.save_many(records)


async def test_list_papers(client: AsyncClient, seeded: list[PaperRecord]) -> None:
    response = await client.get("/library/papers")

    assert response.status_code == 200
    assert len(response.json()) == 4


async def test_list_papers_filtered_by_source(
    client: AsyncClient, seeded: list[PaperRecord]
) -> None:
    response = await client.get("/library/papers", params={"source": "manual"})

    body = response.json()
    assert len(body) == 3
    assert all(record["paper"]["provider"] == "manual" for record in body)


async def test_get_single_paper(client: AsyncClient, seeded: list[PaperRecord]) -> None:
    response = await client.get("/library/papers/manual:01")

    assert response.status_code == 200
    assert response.json()["paper"]["id"] == "manual:01"


async def test_unknown_paper_is_404(client: AsyncClient, seeded: list[PaperRecord]) -> None:
    response = await client.get("/library/papers/manual:999")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "paper_not_found"


async def test_summary_counts_by_source(client: AsyncClient, seeded: list[PaperRecord]) -> None:
    response = await client.get("/library/summary")

    body = response.json()
    assert body["total"] == 4
    assert body["by_source"] == {"manual": 3, "arxiv": 1}
    assert body["pending_parse"] == 4


async def test_source_health(client: AsyncClient) -> None:
    response = await client.get("/library/sources")

    body = response.json()
    assert body[0]["source"] == "manual"
    assert body[0]["healthy"] is True


async def test_retrieve_skips_papers_already_on_disk(
    client: AsyncClient, seeded: list[PaperRecord]
) -> None:
    """Manual papers need no download; the collection is never copied."""
    response = await client.post("/library/retrieve", json={"paper_ids": ["manual:01"]})

    assert response.status_code == 200
    outcome = response.json()["outcomes"][0]
    assert outcome["downloaded"] is False
    assert outcome["reason"] == "already_local"


async def test_retrieve_unknown_papers_is_404(
    client: AsyncClient, seeded: list[PaperRecord]
) -> None:
    response = await client.post("/library/retrieve", json={"paper_ids": ["nope:1"]})

    assert response.status_code == 404


async def test_discovery_run_populates_the_library(
    client: AsyncClient, paper_repository: JsonPaperRepository
) -> None:
    """End to end: plan -> discovery over the real manual collection -> persisted records."""
    response = await client.post(
        "/research/plan", json={"goal": "Metastable failures in distributed systems"}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert body["discovery"]["sources_queried"] == ["manual"]
    assert body["discovery"]["candidates"] > 0
    assert body["candidates"][0]["score"] > 0
    assert "signals" in body["candidates"][0]

    stored = await paper_repository.list_all()
    assert stored
    assert all(record.paper.provider is SourceName.MANUAL for record in stored)
    assert all(record.processing.downloaded for record in stored)
