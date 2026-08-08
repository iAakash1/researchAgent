"""Paper deduplication.

Five indexes describing overlapping literature return the same work repeatedly, in
different shapes: arXiv has the preprint, Crossref the published version, OpenAlex both.
Duplicates would inflate every downstream count and make the v0.8 synthesis claim that
three papers agree when it is one paper seen three times.

Matching is a priority chain, strongest evidence first::

    DOI  ->  arXiv id  ->  other shared identifier  ->  title similarity

Duplicates are *merged*, not discarded: the surviving record keeps the richest metadata
from every copy, so a Crossref DOI and an arXiv PDF link end up on one paper.
"""

from __future__ import annotations

from difflib import SequenceMatcher

from pydantic import BaseModel

from researchagent.config.schemas import DeduplicationConfig
from researchagent.core.logging import get_logger
from researchagent.models.paper import Paper, PaperIdentifiers

logger = get_logger(__name__)


class DeduplicationResult(BaseModel):
    papers: list[Paper]
    duplicates_removed: int = 0
    merged_identifier_count: int = 0

    @property
    def total(self) -> int:
        return len(self.papers)


class PaperDeduplicator:
    """Collapses papers that describe the same work."""

    def __init__(self, config: DeduplicationConfig | None = None) -> None:
        self._config = config or DeduplicationConfig()

    def deduplicate(self, papers: list[Paper]) -> DeduplicationResult:
        survivors: list[Paper] = []
        by_identifier: dict[str, int] = {}
        removed = 0
        merged_ids = 0

        for paper in papers:
            index = self._find_match(paper, survivors, by_identifier)
            if index is None:
                survivors.append(paper)
                self._index(paper, len(survivors) - 1, by_identifier)
                continue

            before = survivors[index].identifiers
            survivors[index] = self._merge(survivors[index], paper)
            if survivors[index].identifiers != before:
                merged_ids += 1
            self._index(survivors[index], index, by_identifier)
            removed += 1

        logger.info(
            "deduplication_complete",
            input=len(papers),
            output=len(survivors),
            removed=removed,
        )
        return DeduplicationResult(
            papers=survivors, duplicates_removed=removed, merged_identifier_count=merged_ids
        )

    def _find_match(
        self, paper: Paper, survivors: list[Paper], by_identifier: dict[str, int]
    ) -> int | None:
        for key in _identifier_keys(paper.identifiers):
            match = by_identifier.get(key)
            if match is not None:
                return match

        if not self._config.compare_titles:
            return None

        candidate_title = paper.normalised_title
        if not candidate_title:
            return None

        for index, survivor in enumerate(survivors):
            # Conflicting identifiers are decisive: two papers with different DOIs are
            # different works no matter how alike their titles read. Without this veto,
            # a series ("... under load", "... under churn") collapses into one paper.
            if _identifiers_conflict(paper.identifiers, survivor.identifiers):
                continue
            if self._titles_match(candidate_title, survivor.normalised_title):
                return index
        return None

    def _titles_match(self, left: str, right: str) -> bool:
        if not left or not right:
            return False
        if left == right:
            return True

        shorter, longer = sorted((len(left), len(right)))
        if longer == 0 or shorter / longer < self._config.length_ratio_floor:
            return False

        ratio = SequenceMatcher(None, left, right).ratio()
        return ratio >= self._config.title_similarity_threshold

    @staticmethod
    def _index(paper: Paper, position: int, by_identifier: dict[str, int]) -> None:
        for key in _identifier_keys(paper.identifiers):
            by_identifier[key] = position

    @staticmethod
    def _merge(keeper: Paper, duplicate: Paper) -> Paper:
        """Combine two views of one paper, preferring whichever copy is richer per field."""
        sources = {
            keeper.provider,
            duplicate.provider,
            *keeper.also_seen_in,
            *duplicate.also_seen_in,
        }
        sources.discard(keeper.provider)

        return keeper.model_copy(
            update={
                "identifiers": keeper.identifiers.merge(duplicate.identifiers),
                "abstract": _longer(keeper.abstract, duplicate.abstract),
                "authors": keeper.authors or duplicate.authors,
                "year": keeper.year or duplicate.year,
                "venue": keeper.venue or duplicate.venue,
                "url": keeper.url or duplicate.url,
                # A retrievable PDF is the scarcest attribute; never lose one.
                "pdf_url": keeper.pdf_url or duplicate.pdf_url,
                "local_path": keeper.local_path or duplicate.local_path,
                "keywords": _union(keeper.keywords, duplicate.keywords),
                "citation_count": _max_optional(keeper.citation_count, duplicate.citation_count),
                "publication_type": (
                    duplicate.publication_type
                    if keeper.publication_type.value == "unknown"
                    else keeper.publication_type
                ),
                "is_open_access": (
                    True
                    if keeper.is_open_access or duplicate.is_open_access
                    else keeper.is_open_access
                ),
                "also_seen_in": sorted(sources, key=lambda source: source.value),
                "source_metadata": {
                    **duplicate.source_metadata,
                    **keeper.source_metadata,
                },
            }
        )


def _identifier_keys(identifiers: PaperIdentifiers) -> list[str]:
    """Ordered match keys. DOI first: it is the only globally authoritative id."""
    keys = []
    for field in (
        "doi",
        "arxiv_id",
        "pubmed_id",
        "openalex_id",
        "semantic_scholar_id",
        "corpus_id",
    ):
        value = getattr(identifiers, field)
        if value:
            keys.append(f"{field}:{value}")
    return keys


def _identifiers_conflict(left: PaperIdentifiers, right: PaperIdentifiers) -> bool:
    """True when both records name the same identifier field with different values."""
    for field in ("doi", "arxiv_id", "pubmed_id", "openalex_id", "semantic_scholar_id"):
        left_value, right_value = getattr(left, field), getattr(right, field)
        if left_value and right_value and left_value != right_value:
            return True
    return False


def _longer(left: str | None, right: str | None) -> str | None:
    if left and right:
        return left if len(left) >= len(right) else right
    return left or right


def _union(left: list[str], right: list[str]) -> list[str]:
    seen: dict[str, str] = {}
    for value in [*left, *right]:
        seen.setdefault(value.lower(), value)
    return list(seen.values())


def _max_optional(left: int | None, right: int | None) -> int | None:
    values = [value for value in (left, right) if value is not None]
    return max(values) if values else None
