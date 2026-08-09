"""Graph API tests.

The endpoints are domain-level by design: there is no route that accepts a query language,
so a caller cannot read the graph in a way the provenance model cannot describe. The first
test pins that.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from httpx import ASGITransport, AsyncClient

from researchagent.api.app import create_app
from researchagent.container import Container
from researchagent.core.validation import Confidence, ValidationResult
from researchagent.models.knowledge import PaperKnowledge
from researchagent.schemas.knowledge import ValidatedKnowledge


@pytest.fixture
async def client(container: Container) -> AsyncIterator[AsyncClient]:
    app = create_app(container=container)
    async with (
        AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as http_client,
        app.router.lifespan_context(app),
    ):
        yield http_client


@pytest.fixture
async def seeded(container: Container, paper_a: PaperKnowledge, paper_b: PaperKnowledge) -> None:
    for knowledge in (paper_a, paper_b):
        await container.knowledge_repository.save(
            ValidatedKnowledge(
                value=knowledge,
                validation=ValidationResult.passed(
                    validator="test",
                    subject_id=knowledge.paper_id,
                    subject_type="PaperKnowledge",
                    confidence=Confidence(score=0.9),
                ),
            )
        )


def test_no_endpoint_accepts_a_raw_query_language(container: Container) -> None:
    app = create_app(container=container)
    graph_paths = [path for path in app.openapi()["paths"] if path.startswith("/graph")]

    assert graph_paths, "the graph router must be mounted"
    assert not any("cypher" in path.lower() or "query" in path.lower() for path in graph_paths)


async def test_querying_before_a_build_says_so_rather_than_returning_empty(
    client: AsyncClient,
) -> None:
    response = await client.get("/graph/stats")

    assert response.status_code == 409
    body = response.json()
    assert body["error"]["code"] == "graph_not_built"
    assert "/graph/build" in body["error"]["remedy"]


async def test_build_then_stats(client: AsyncClient, seeded: None) -> None:
    build = await client.post("/graph/build", json={})

    assert build.status_code == 200
    report = build.json()
    assert report["succeeded"] is True
    assert report["papers"] == 2
    assert report["provenance_coverage"] == 1.0

    stats = await client.get("/graph/stats")
    assert stats.status_code == 200
    assert stats.json()["nodes"] == report["nodes"]


async def test_rebuilding_is_idempotent(client: AsyncClient, seeded: None) -> None:
    first = (await client.post("/graph/build", json={})).json()
    second = (await client.post("/graph/build", json={})).json()

    assert first["version"] == second["version"]
    assert (first["nodes"], first["edges_accepted"]) == (second["nodes"], second["edges_accepted"])

    versions = (await client.get("/graph/versions")).json()
    assert len(versions["versions"]) == 1, "a rebuild replaces a generation, it does not add one"


async def test_datasets_for_a_method_come_back_with_citations(
    client: AsyncClient, seeded: None
) -> None:
    await client.post("/graph/build", json={})

    response = await client.get("/graph/methods/RAG/datasets")

    assert response.status_code == 200
    found = response.json()
    assert [entity["node"]["name"] for entity in found] == ["MIMIC-III"]
    assert found[0]["citations"], "an answer without a citation is an assertion"


async def test_metrics_for_a_method(client: AsyncClient, seeded: None) -> None:
    await client.post("/graph/build", json={})

    response = await client.get("/graph/methods/RAG/metrics")

    assert [entity["node"]["name"] for entity in response.json()] == ["F1"]


async def test_papers_using_an_entity(client: AsyncClient, seeded: None) -> None:
    await client.post("/graph/build", json={})

    response = await client.get("/graph/entities/MIMIC-III/papers")

    assert len(response.json()) == 2


async def test_shared_entities_across_papers(client: AsyncClient, seeded: None) -> None:
    await client.post("/graph/build", json={})

    response = await client.get("/graph/shared", params={"kind": "Dataset"})

    shared = response.json()
    assert shared
    assert shared[0]["node"]["name"] == "MIMIC-III"


async def test_contradictions_return_both_sides(client: AsyncClient, seeded: None) -> None:
    await client.post("/graph/build", json={})

    response = await client.get("/graph/contradictions")

    pairs = response.json()
    assert pairs, "0.82 vs 0.41 on the same metric and dataset"
    assert pairs[0]["left"]["id"] != pairs[0]["right"]["id"]
    assert len(set(pairs[0]["papers"])) == 2


async def test_provenance_for_a_relationship(client: AsyncClient, seeded: None) -> None:
    await client.post("/graph/build", json={})
    datasets = (await client.get("/graph/methods/RAG/datasets")).json()
    dataset_id = datasets[0]["node"]["id"]
    methods = (await client.get("/graph/datasets/MIMIC-III/methods")).json()
    method_id = methods[0]["node"]["id"]

    response = await client.get(
        "/graph/provenance", params={"source_id": method_id, "target_id": dataset_id}
    )

    body = response.json()
    assert body["grounded"] is True
    assert body["citations"]


async def test_neighbours_returns_a_subgraph(client: AsyncClient, seeded: None) -> None:
    await client.post("/graph/build", json={})
    method_id = (await client.get("/graph/datasets/MIMIC-III/methods")).json()[0]["node"]["id"]

    response = await client.get(f"/graph/nodes/{method_id}/neighbours", params={"depth": 1})

    body = response.json()
    assert body["centre_id"] == method_id
    assert body["nodes"] and body["edges"]


async def test_an_unknown_method_returns_an_empty_list_not_an_error(
    client: AsyncClient, seeded: None
) -> None:
    await client.post("/graph/build", json={})

    response = await client.get("/graph/methods/NotAMethod/datasets")

    assert response.status_code == 200
    assert response.json() == []
