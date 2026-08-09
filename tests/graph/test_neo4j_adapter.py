"""Neo4j adapter tests.

A fake driver stands in for the real one, so these run offline and the suite never needs a
Neo4j server. What is under test is the adapter's contract — that it MERGEs on domain ids
(which is what makes construction idempotent), scopes everything by graph version, reports
a dead server rather than raising from health(), and translates driver failures into the
project's error taxonomy.
"""

from __future__ import annotations

from typing import Any

import pytest

from researchagent.core.exceptions import GraphStoreError
from researchagent.integrations.neo4j import Neo4jGraphRepository
from researchagent.models.graph import GraphVersion


class FakeResult:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows

    def __aiter__(self) -> FakeResult:
        self._iter = iter(self._rows)
        return self

    async def __anext__(self) -> Any:
        try:
            return FakeRecord(next(self._iter))
        except StopIteration:
            raise StopAsyncIteration from None


class FakeRecord:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def data(self) -> dict[str, Any]:
        return self._payload


class FakeSession:
    def __init__(self, driver: FakeDriver) -> None:
        self._driver = driver

    async def __aenter__(self) -> FakeSession:
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    async def run(self, cypher: str, **params: Any) -> FakeResult:
        self._driver.queries.append((cypher, params))
        if self._driver.error is not None:
            raise self._driver.error
        for pattern, rows in self._driver.responses.items():
            if pattern in cypher:
                return FakeResult(rows)
        return FakeResult([])


class FakeDriver:
    def __init__(
        self,
        responses: dict[str, list[dict[str, Any]]] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.queries: list[tuple[str, dict[str, Any]]] = []
        self.responses = responses or {}
        self.error = error
        self.closed = False

    def session(self, **_: Any) -> FakeSession:
        return FakeSession(self)

    async def close(self) -> None:
        self.closed = True


@pytest.fixture
def version() -> GraphVersion:
    return GraphVersion(corpus_fingerprint="abc123def456")


def repository(driver: FakeDriver) -> Neo4jGraphRepository:
    repo = Neo4jGraphRepository(uri="bolt://unused:7687", password="unused")  # noqa: S106
    repo._driver = driver
    return repo


async def test_writing_merges_on_domain_ids_so_rebuilds_do_not_duplicate(
    paper_a: object, version: GraphVersion
) -> None:
    from tests.graph.test_graph import build

    driver = FakeDriver()
    graph = build([paper_a])  # type: ignore[list-item]

    await repository(driver).write_graph(graph)

    cyphers = " ".join(cypher for cypher, _ in driver.queries)
    assert "MERGE (n:RAEntity {id: row.id" in cyphers, "identity is the domain id"
    assert "MERGE (a)-[r:RELATES {id: row.id" in cyphers
    assert "CREATE" not in cyphers.replace("CREATE CONSTRAINT", ""), "MERGE, never CREATE"


async def test_writing_declares_a_uniqueness_constraint_on_id_and_version(
    paper_a: object,
) -> None:
    from tests.graph.test_graph import build

    driver = FakeDriver()
    await repository(driver).write_graph(build([paper_a]))  # type: ignore[list-item]

    constraint = next(c for c, _ in driver.queries if "CONSTRAINT" in c)
    assert "(n.id, n.graph_version) IS UNIQUE" in constraint


async def test_every_query_is_scoped_to_one_generation(version: GraphVersion) -> None:
    """Two graph generations must never be readable as one."""
    driver = FakeDriver()
    repo = repository(driver)

    await repo.get_node("method:rag:x", version)
    await repo.find_nodes(version=version)
    await repo.stats(version)

    for cypher, params in driver.queries:
        assert "graph_version" in cypher
        assert params.get("version") == version.identifier


async def test_node_properties_round_trip_through_storage(version: GraphVersion) -> None:
    driver = FakeDriver(
        responses={
            "RETURN n LIMIT 1": [
                {
                    "n": {
                        "id": "dataset:mimic-iii:abc",
                        "kind": "Dataset",
                        "name": "MIMIC-III",
                        "paper_ids": ["manual:01", "manual:02"],
                        "evidence_ids": ["e1"],
                        "confidence": 0.8,
                    }
                }
            ]
        }
    )

    node = await repository(driver).get_node("dataset:mimic-iii:abc", version)

    assert node is not None
    assert node.name == "MIMIC-III"
    assert node.paper_ids == ("manual:01", "manual:02")
    assert node.confidence.score == pytest.approx(0.8)


async def test_stats_aggregate_by_kind(version: GraphVersion) -> None:
    driver = FakeDriver(
        responses={
            "RETURN n.kind AS kind": [
                {"kind": "Method", "total": 3},
                {"kind": "Dataset", "total": 2},
            ],
            "RETURN r.kind AS kind": [{"kind": "EVALUATED_ON", "total": 4}],
        }
    )

    stats = await repository(driver).stats(version)

    assert stats.nodes == 5
    assert stats.edges == 4
    assert stats.nodes_by_kind == {"Method": 3, "Dataset": 2}


async def test_depth_is_clamped_rather_than_interpolated_freely(version: GraphVersion) -> None:
    """Depth reaches Cypher as a literal, so it must be bounded."""
    driver = FakeDriver()

    await repository(driver).neighbours("method:rag:x", version, depth=99)

    cypher = next(c for c, _ in driver.queries if "RELATES*" in c)
    assert "RELATES*1..5" in cypher


async def test_a_driver_failure_becomes_a_recoverable_graph_store_error(
    version: GraphVersion,
) -> None:
    """The graph is derived, so an unreachable Neo4j is never fatal."""
    driver = FakeDriver(error=RuntimeError("connection refused"))

    with pytest.raises(GraphStoreError) as caught:
        await repository(driver).stats(version)

    assert caught.value.recoverability.allows_continue
    assert "rebuilt" in (caught.value.remedy or "")


async def test_health_reports_false_instead_of_raising() -> None:
    driver = FakeDriver(error=RuntimeError("no route to host"))

    assert await repository(driver).health() is False


async def test_health_reports_true_when_reachable() -> None:
    driver = FakeDriver(responses={"RETURN 1 AS ok": [{"ok": 1}]})

    assert await repository(driver).health() is True


async def test_closing_releases_the_driver() -> None:
    driver = FakeDriver()
    repo = repository(driver)

    await repo.aclose()

    assert driver.closed
    assert repo._driver is None


def test_the_adapter_is_the_only_place_neo4j_is_imported() -> None:
    """Architectural guard: the domain must not depend on a vendor SDK."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[2] / "researchagent"
    offenders = [
        path.relative_to(root).as_posix()
        for path in root.rglob("*.py")
        if "neo4j" in path.read_text().lower()
        and "import" in path.read_text()
        and path.parent.name != "neo4j"
        and _imports_driver(path.read_text())
    ]

    assert offenders == [], f"neo4j driver imported outside the adapter: {offenders}"


def _imports_driver(source: str) -> bool:
    return "from neo4j import" in source or "import neo4j" in source
