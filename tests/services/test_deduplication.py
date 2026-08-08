from __future__ import annotations

from pathlib import Path

from researchagent.config.schemas import DeduplicationConfig
from researchagent.models.paper import Paper, PaperIdentifiers, PublicationType, SourceName
from researchagent.services.deduplication import PaperDeduplicator


def paper(
    title: str = "Metastable failures in distributed systems",
    provider: SourceName = SourceName.ARXIV,
    **overrides: object,
) -> Paper:
    return Paper.model_validate(
        {"id": f"{provider.value}:{title[:12]}", "title": title, "provider": provider} | overrides
    )


def test_identical_dois_are_merged() -> None:
    left = paper(provider=SourceName.CROSSREF, identifiers=PaperIdentifiers(doi="10.1145/123"))
    right = paper(
        title="A completely different sounding title",
        provider=SourceName.OPENALEX,
        identifiers=PaperIdentifiers(doi="10.1145/123"),
    )

    result = PaperDeduplicator().deduplicate([left, right])

    assert result.total == 1
    assert result.duplicates_removed == 1


def test_doi_matching_survives_url_and_case_differences() -> None:
    left = paper(identifiers=PaperIdentifiers(doi="https://doi.org/10.1145/ABC"))
    right = paper(title="Another title", identifiers=PaperIdentifiers(doi="10.1145/abc"))

    assert PaperDeduplicator().deduplicate([left, right]).total == 1


def test_arxiv_versions_are_the_same_work() -> None:
    left = paper(identifiers=PaperIdentifiers(arxiv_id="2401.12345v2"))
    right = paper(
        title="Different title entirely here",
        identifiers=PaperIdentifiers(arxiv_id="arXiv:2401.12345"),
    )

    assert PaperDeduplicator().deduplicate([left, right]).total == 1


def test_near_identical_titles_are_merged() -> None:
    left = paper("Metastable Failures in Distributed Systems")
    right = paper("Metastable failures in distributed systems.", provider=SourceName.OPENALEX)

    result = PaperDeduplicator().deduplicate([left, right])

    assert result.total == 1
    assert result.papers[0].also_seen_in == [SourceName.OPENALEX]


def test_genuinely_different_papers_are_kept() -> None:
    papers = [
        paper("Metastable failures in distributed systems"),
        paper("Congestion avoidance and control"),
        paper("Why do multi-agent LLM systems fail"),
    ]

    assert PaperDeduplicator().deduplicate(papers).total == 3


def test_short_title_is_not_matched_against_a_long_one() -> None:
    left = paper("Agents")
    right = paper("Agents for automated clinical decision support in emergency triage")

    assert PaperDeduplicator().deduplicate([left, right]).total == 2


def test_merge_keeps_the_richest_metadata() -> None:
    sparse = paper(
        provider=SourceName.ARXIV,
        identifiers=PaperIdentifiers(arxiv_id="2401.1"),
        abstract="Short.",
        pdf_url="https://arxiv.org/pdf/2401.1",
        keywords=["distributed systems"],
    )
    rich = paper(
        provider=SourceName.OPENALEX,
        identifiers=PaperIdentifiers(arxiv_id="2401.1", doi="10.1145/999"),
        abstract="A considerably longer and more useful abstract than the other one.",
        year=2024,
        venue="SOSP",
        citation_count=42,
        keywords=["metastability"],
        publication_type=PublicationType.CONFERENCE_PAPER,
    )

    merged = PaperDeduplicator().deduplicate([sparse, rich]).papers[0]

    assert merged.identifiers.doi == "10.1145/999"  # gained from the duplicate
    assert merged.identifiers.arxiv_id == "2401.1"
    assert merged.abstract is not None
    assert merged.abstract.startswith("A considerably longer")
    assert merged.year == 2024
    assert merged.venue == "SOSP"
    assert merged.citation_count == 42
    assert merged.publication_type is PublicationType.CONFERENCE_PAPER
    assert set(merged.keywords) == {"distributed systems", "metastability"}
    # The scarcest attribute of all — a retrievable PDF — must never be lost in a merge.
    assert merged.pdf_url == "https://arxiv.org/pdf/2401.1"


def test_merge_preserves_a_local_path(tmp_path: Path) -> None:
    local = paper(provider=SourceName.MANUAL, local_path=tmp_path / "01.pdf")
    remote = paper(provider=SourceName.ARXIV, pdf_url="https://arxiv.org/pdf/1")

    merged = PaperDeduplicator().deduplicate([local, remote]).papers[0]

    assert merged.local_path == tmp_path / "01.pdf"
    assert merged.pdf_url == "https://arxiv.org/pdf/1"


def test_three_way_duplicate_collapses_to_one() -> None:
    doi = PaperIdentifiers(doi="10.1/x")
    papers = [
        paper(provider=SourceName.ARXIV, identifiers=doi),
        paper(provider=SourceName.CROSSREF, identifiers=doi),
        paper(provider=SourceName.OPENALEX, identifiers=doi),
    ]

    result = PaperDeduplicator().deduplicate(papers)

    assert result.total == 1
    assert result.duplicates_removed == 2
    assert result.papers[0].also_seen_in == [SourceName.CROSSREF, SourceName.OPENALEX]


def test_title_matching_can_be_disabled() -> None:
    config = DeduplicationConfig(compare_titles=False)
    papers = [paper("Same title here"), paper("Same title here", provider=SourceName.OPENALEX)]

    assert PaperDeduplicator(config).deduplicate(papers).total == 2


def test_empty_input_is_handled() -> None:
    assert PaperDeduplicator().deduplicate([]).total == 0


def test_conflicting_dois_veto_a_title_match() -> None:
    """A paper series shares almost all of its title but is not one paper."""
    first = paper("Metastable failures under load", identifiers=PaperIdentifiers(doi="10.1/a"))
    second = paper(
        "Metastable failures under churn",
        provider=SourceName.OPENALEX,
        identifiers=PaperIdentifiers(doi="10.1/b"),
    )

    assert PaperDeduplicator().deduplicate([first, second]).total == 2


def test_a_missing_identifier_does_not_veto() -> None:
    with_doi = paper(identifiers=PaperIdentifiers(doi="10.1/a"))
    without = paper(provider=SourceName.MANUAL)

    result = PaperDeduplicator().deduplicate([with_doi, without])

    assert result.total == 1
    assert result.papers[0].identifiers.doi == "10.1/a"
