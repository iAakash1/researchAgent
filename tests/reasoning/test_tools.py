"""Tool argument validation and the call ledger.

The toolbox is the only I/O an agent may perform, so what it refuses matters as much as
what it does.
"""

from __future__ import annotations

import pytest

from researchagent.container import Container
from researchagent.core.interfaces.tools import ToolName
from researchagent.services.tools.toolbox import (
    MAX_DEPTH,
    MAX_LIMIT,
    MAX_QUERY_CHARS,
    _clamp,
    _clean_query,
    _parse_kinds,
)


class TestArgumentValidation:
    @pytest.mark.parametrize(
        ("asked", "expected"), [(1, 1), (10, 10), (999, MAX_LIMIT), (0, 1), (-5, 1)]
    )
    def test_limits_are_clamped_never_honoured_blindly(self, asked: int, expected: int) -> None:
        """A model-supplied limit is a suggestion, not an instruction."""
        assert _clamp(asked, MAX_LIMIT) == expected

    def test_depth_is_clamped_separately(self) -> None:
        assert _clamp(99, MAX_DEPTH) == MAX_DEPTH

    def test_an_empty_query_is_refused(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            _clean_query("   ")

    def test_a_long_query_is_truncated_not_rejected(self) -> None:
        cleaned = _clean_query("overload " * 500)

        assert len(cleaned) <= MAX_QUERY_CHARS

    def test_whitespace_is_normalised(self) -> None:
        assert _clean_query("  circuit\n\tbreaker  ") == "circuit breaker"

    def test_unknown_kinds_are_dropped_not_guessed(self) -> None:
        """Dropping gives an unfiltered search, which is the safer failure."""
        parsed = _parse_kinds(("method", "benchmark", "DATASET"))

        assert [kind.value for kind in parsed] == ["method", "dataset"]

    def test_no_kinds_means_no_filter(self) -> None:
        assert _parse_kinds(()) == ()


class TestToolLedger:
    async def test_every_call_is_recorded_with_its_agent(self, container: Container) -> None:
        toolbox = container.toolbox.for_agent("retrieval", 0)

        await toolbox.search_knowledge("overload mitigation")

        assert toolbox.calls
        call = toolbox.calls[-1]
        assert call.tool is ToolName.SEARCH_KNOWLEDGE
        assert call.agent == "retrieval"
        assert call.latency_ms >= 0.0

    async def test_agent_views_share_one_ledger(self, container: Container) -> None:
        """Attribution per agent, one audit trail per run."""
        retrieval = container.toolbox.for_agent("retrieval", 0)
        verification = container.toolbox.for_agent("verification", 0)

        await retrieval.search_knowledge("overload")
        await verification.get_provenance(("nonexistent",))

        agents = {call.agent for call in retrieval.calls}
        assert agents == {"retrieval", "verification"}

    async def test_an_empty_result_is_recorded_not_raised(self, container: Container) -> None:
        """ "Nothing found" is an answer an agent must be able to act on."""
        toolbox = container.toolbox.for_agent("retrieval", 0)

        result = await toolbox.search_knowledge("a query matching nothing at all xyzzy")

        assert result.objects == () or result.objects
        assert toolbox.calls[-1].succeeded

    async def test_provenance_of_unknown_ids_returns_empty(self, container: Container) -> None:
        toolbox = container.toolbox.for_agent("verification", 0)

        assert await toolbox.get_provenance(("no-such-evidence",)) == ()

    async def test_graph_search_reports_unavailable_rather_than_failing(
        self, container: Container
    ) -> None:
        """No graph built yet is a state to report, not an exception to handle."""
        toolbox = container.toolbox.for_agent("retrieval", 0)

        result = await toolbox.search_graph("Circuit Breaker")

        assert result.available is False
        assert result.nodes == ()

    async def test_paper_context_of_an_unknown_paper_is_reported_not_raised(
        self, container: Container
    ) -> None:
        toolbox = container.toolbox.for_agent("reasoning", 0)

        context = await toolbox.get_paper_context("manual:does-not-exist")

        assert context.found is False


class TestToolSurface:
    def test_there_is_no_tool_that_accepts_a_query_language(self) -> None:
        """No execute(cypher), no search(sql), no eval."""
        names = {tool.value for tool in ToolName}

        assert not any(
            word in name for name in names for word in ("execute", "eval", "sql", "cypher", "raw")
        )

    def test_the_vocabulary_is_closed(self) -> None:
        """An agent naming anything outside this set is rejected, not guessed at."""
        assert {tool.value for tool in ToolName} == {
            "search_knowledge",
            "retrieve_evidence",
            "build_bundle",
            "search_graph",
            "get_provenance",
            "find_contradictions",
            "get_paper_context",
        }


class TestBundlePersistence:
    async def test_a_built_bundle_is_retrievable_by_its_id(self, container: Container) -> None:
        """Findings cite bundle ids; an id that cannot be loaded back is a dead-ended audit.

        Caught by the first real research run, where reasoning silently produced nothing
        because every bundle the toolbox built had already been thrown away.
        """
        toolbox = container.toolbox.for_agent("retrieval", 0)

        bundle = await toolbox.build_bundle("overload mitigation techniques")
        loaded = await container.bundle_repository.get(bundle.id)

        assert loaded is not None
        assert loaded.id == bundle.id
