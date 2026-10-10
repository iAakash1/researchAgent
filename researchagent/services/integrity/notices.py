"""Secondary integrity signal: retraction or concern wording in a parsed PDF.

Publishers stamp retracted PDFs ("This article has been retracted"). Finding such text is
worth a human look, but it is weak evidence — the phrase can appear in a paper *about*
retractions — so a match only ever produces ``UNKNOWN`` flagged for review. It never sets
``CURRENT`` and never invalidates a conclusion by itself.
"""

from __future__ import annotations

from researchagent.config.schemas import PdfNoticeSettings
from researchagent.models.document import PaperDocument
from researchagent.utils.text import normalise


class RetractionNoticeScanner:
    name = "pdf_notice_scanner"

    def __init__(self, settings: PdfNoticeSettings) -> None:
        self._settings = settings
        self._patterns = tuple(normalise(pattern) for pattern in settings.patterns if pattern)

    @property
    def enabled(self) -> bool:
        return self._settings.enabled and bool(self._patterns)

    def scan(self, document: PaperDocument) -> str | None:
        """The first configured phrase found in the document's opening text, or None."""
        if not self.enabled:
            return None
        title = document.metadata.title or ""
        text = normalise(f"{title}\n{document.full_text}"[: self._settings.max_chars])
        return next((pattern for pattern in self._patterns if pattern in text), None)
