"""Document loading port.

Splits cleanly in two so that no vendor library leaks into domain logic:

* This port yields a :class:`RawDocument` — pages of positioned text blocks with font
  information. That is the only thing PyMuPDF (or any replacement) is responsible for.
* Section detection, reference extraction and assembly are pure functions over that
  structure and live in ``services/document/``.

Swapping PyMuPDF for pdfplumber, Grobid or a layout model is therefore a new adapter,
with no change to detection, validation or the canonical document model.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar

from researchagent.models.layout import RawDocument


class DocumentLoader(ABC):
    """Turns a PDF file into positioned, font-annotated text blocks."""

    name: ClassVar[str]
    version: ClassVar[str]

    @abstractmethod
    def load(self, path: Path, *, document_id: str) -> RawDocument:
        """Read ``path``.

        Raises ``DocumentUnreadableError`` when the file cannot be opened and
        ``DocumentParsingError`` when it opens but yields no usable structure.
        """

    @abstractmethod
    def supports(self, path: Path) -> bool:
        """Whether this loader handles the file type."""
