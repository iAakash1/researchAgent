from __future__ import annotations

from researchagent.models.library import ProcessingStatus


def test_stage_reached_returns_furthest_stage() -> None:
    # None of the flags are set initially (by default discovered is True,
    # but we can test explicit states)
    status = ProcessingStatus(discovered=False)
    assert status.stage_reached == "pending"

    # Set one early flag
    status = ProcessingStatus(discovered=True, downloaded=False)
    assert status.stage_reached == "discovered"

    # Set multiple flags
    status = ProcessingStatus(discovered=True, downloaded=True, parsed=True)
    assert status.stage_reached == "parsed"

    # Set non-contiguous flags, it should return the one furthest along the pipeline
    status = ProcessingStatus(discovered=True, parsed=True, embedded=True)
    assert status.stage_reached == "embedded"

    # Set all flags
    status = ProcessingStatus(
        discovered=True,
        downloaded=True,
        validated=True,
        parsed=True,
        sectioned=True,
        references_extracted=True,
        figures_extracted=True,
        tables_extracted=True,
        ready_for_extraction=True,
        extracted=True,
        chunked=True,
        embedded=True,
        verified=True,
        indexed_in_graph=True,
    )
    assert status.stage_reached == "indexed_in_graph"
