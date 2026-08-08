"""Manual library adapter.

Treats a directory of hand-collected PDFs as a first-class literature source, so curated
papers compete in discovery on the same terms as anything from arXiv.

The filename convention already in use carries real metadata::

    01_[P1]_Metastable_Failures_in_Distributed_Systems.pdf
    05a_[P1]_MCP_Specification_2026-07-28_Overview.pdf
     |   |    |
     |   |    title (underscores -> spaces)
     |   priority tag
     index (may carry a letter suffix for split documents)

Anything the filename does not state is left empty. In particular a bare four-digit
number is *not* read as a year — ``09_[P3]_A2A_Issue_1987_Idempotency`` would otherwise
be dated 1987. Only a full ISO date is trusted. Never invent metadata.

Files are read, never moved, renamed or modified.
"""

from __future__ import annotations

import re
from pathlib import Path

from researchagent.core.exceptions import PaperNotFoundError
from researchagent.core.interfaces.paper_source import PaperSource, SearchQuery, SourceHealth
from researchagent.core.logging import get_logger
from researchagent.models.paper import (
    Paper,
    PaperIdentifiers,
    PublicationType,
    SourceName,
    make_paper_id,
    normalise_title,
)

logger = get_logger(__name__)

_FILENAME = re.compile(
    r"^(?P<index>\d+[a-z]?)"  # 01, 05a
    r"(?:_\[(?P<priority>P\d)\])?"  # optional [P1]
    r"_(?P<title>.+)$"
)
_ISO_DATE = re.compile(r"(?P<year>19\d{2}|20\d{2})-\d{2}-\d{2}")
_STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for",
        "from", "how", "in", "into", "is", "it", "its", "of", "on", "or", "our", "that",
        "the", "their", "there", "these", "this", "those", "to", "via", "we", "what",
        "which", "why", "with",
    }
)  # fmt: skip


class ManualPaperSource(PaperSource):
    """Scans a directory of PDFs and exposes them through the PaperSource port."""

    name = SourceName.MANUAL

    def __init__(self, library_dir: Path) -> None:
        self._library_dir = library_dir

    @property
    def library_dir(self) -> Path:
        return self._library_dir

    async def search(self, query: SearchQuery) -> list[Paper]:
        """Token-overlap search over filename-derived titles.

        A local collection is small enough that lexical matching is honest and adequate;
        semantic search over this corpus arrives with the RAG work.
        """
        wanted = _tokenise(" ".join([query.text, *query.terms]))
        if not wanted:
            return []

        scored: list[tuple[int, Paper]] = []
        for paper in self.load_all():
            overlap = len(wanted & _tokenise(paper.title))
            if overlap:
                scored.append((overlap, paper))

        scored.sort(key=lambda pair: (-pair[0], pair[1].title))
        papers = [paper for _, paper in scored[: query.limit]]
        logger.debug("manual_search", query=query.text, matched=len(papers), scanned=len(scored))
        return papers

    async def get_paper(self, identifier: str) -> Paper | None:
        wanted = identifier.removeprefix("manual:")
        return next((p for p in self.load_all() if p.source_metadata.get("index") == wanted), None)

    async def download_pdf(self, paper: Paper, destination: Path) -> Path:
        """No-op: the file is already on disk.

        Returns its existing path rather than copying, because the manual library is
        the user's own collection and must not be duplicated or relocated.
        """
        if paper.local_path is None or not paper.local_path.is_file():
            raise PaperNotFoundError(
                "Manual paper file is missing", paper_id=paper.id, source=self.name.value
            )
        return paper.local_path

    async def health(self) -> SourceHealth:
        if not self._library_dir.is_dir():
            return SourceHealth(
                source=self.name,
                healthy=False,
                detail=f"Library directory not found: {self._library_dir}",
            )
        count = len(list(self._library_dir.glob("*.pdf")))
        return SourceHealth(source=self.name, healthy=True, detail=f"{count} local papers")

    async def aclose(self) -> None:
        return None

    def load_all(self) -> list[Paper]:
        """Every PDF in the library, newest scan each call so added files appear at once."""
        if not self._library_dir.is_dir():
            logger.warning("manual_library_missing", directory=str(self._library_dir))
            return []
        return [self._to_paper(path) for path in sorted(self._library_dir.glob("*.pdf"))]

    def _to_paper(self, path: Path) -> Paper:
        match = _FILENAME.match(path.stem)
        if match is None:
            # Unconventional filename: use the stem as the title rather than skipping the
            # file, so nothing in the user's collection silently disappears.
            index, priority, raw_title = path.stem, None, path.stem
        else:
            index = match.group("index")
            priority = match.group("priority")
            raw_title = match.group("title")

        title = raw_title.replace("_", " ").strip()
        identifiers = PaperIdentifiers()

        return Paper(
            id=f"manual:{index}" if match else make_paper_id(identifiers, self.name, title),
            title=title,
            abstract=None,
            year=_year_from_filename(path.stem),
            venue=None,
            identifiers=identifiers,
            url=path.as_uri(),
            pdf_url=None,
            local_path=path,
            provider=self.name,
            publication_type=PublicationType.UNKNOWN,
            is_open_access=True,
            source_metadata={
                "index": index,
                "priority": priority,
                "filename": path.name,
                "size_bytes": path.stat().st_size,
            },
        )


def _year_from_filename(stem: str) -> int | None:
    """Only trust a full ISO date. A bare four-digit number is far more often an issue
    number, a version or part of a title than a publication year."""
    match = _ISO_DATE.search(stem)
    return int(match.group("year")) if match else None


def _tokenise(text: str) -> set[str]:
    return {
        token
        for token in normalise_title(text).split()
        if len(token) > 2 and token not in _STOPWORDS
    }
