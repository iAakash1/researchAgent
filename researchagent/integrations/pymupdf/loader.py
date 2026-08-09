"""PyMuPDF document loader.

The single place in the codebase that imports a PDF library. It produces positioned,
font-annotated text blocks and stops there — it makes no judgement about what a block
*means*, because section detection tested through a PDF fixture is section detection
nobody can debug.

Assumes digital PDFs. A scanned document surfaces as pages with no extractable text,
which the PDF validator reports as a fatal issue with an OCR remedy rather than silently
producing an empty document.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, ClassVar

import pymupdf

from researchagent.core.evidence import BoundingBox
from researchagent.core.exceptions import DocumentParsingError, DocumentUnreadableError
from researchagent.core.interfaces.document_parser import DocumentLoader
from researchagent.core.logging import get_logger
from researchagent.models.layout import (
    PdfMetadata,
    RawDocument,
    RawPage,
    TextBlock,
    TextStyle,
)

logger = get_logger(__name__)

# PyMuPDF span flag bits.
_FLAG_ITALIC = 1 << 1
_FLAG_BOLD = 1 << 4

_TEXT_BLOCK = 0
_HASH_CHUNK_BYTES = 1 << 20


class PyMuPDFLoader(DocumentLoader):
    name: ClassVar[str] = "pymupdf"
    version: ClassVar[str] = pymupdf.__version__ if hasattr(pymupdf, "__version__") else ""

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() == ".pdf"

    def load(self, path: Path, *, document_id: str) -> RawDocument:
        if not path.is_file():
            raise DocumentUnreadableError(
                "PDF file does not exist", paper_id=document_id, path=str(path)
            )

        size = path.stat().st_size
        if size == 0:
            raise DocumentUnreadableError("PDF file is empty", paper_id=document_id, path=str(path))

        try:
            document = pymupdf.open(path)  # type: ignore[no-untyped-call]
        except Exception as exc:
            # PyMuPDF raises a variety of native errors; none of them should escape the
            # integration boundary untyped.
            raise DocumentUnreadableError(
                "PDF could not be opened",
                paper_id=document_id,
                path=str(path),
                reason=str(exc),
                error_type=type(exc).__name__,
            ) from exc

        try:
            if document.needs_pass:
                raise DocumentUnreadableError(
                    "PDF is password protected", paper_id=document_id, path=str(path)
                )
            if document.page_count == 0:
                raise DocumentParsingError(
                    "PDF contains no pages", paper_id=document_id, path=str(path)
                )

            pages = tuple(
                self._read_page(document, number, document_id)
                for number in range(document.page_count)
            )
            metadata = _metadata_of(document)
        finally:
            document.close()  # type: ignore[no-untyped-call]

        raw = RawDocument(
            document_id=document_id,
            pages=pages,
            pdf_metadata=metadata,
            source_sha256=_sha256(path),
            source_bytes=size,
            loader=self.name,
            loader_version=self.version,
        )
        logger.debug(
            "pdf_loaded",
            paper_id=document_id,
            pages=raw.page_count,
            blocks=len(raw.blocks),
            characters=raw.character_count,
            empty_pages=raw.empty_page_count,
        )
        return raw

    def _read_page(self, document: Any, index: int, document_id: str) -> RawPage:
        page = document[index]
        try:
            content = page.get_text("dict")
        except Exception as exc:
            raise DocumentParsingError(
                "Page text could not be extracted",
                paper_id=document_id,
                page=index + 1,
                reason=str(exc),
            ) from exc

        blocks = []
        order = 0
        for block in content.get("blocks", []):
            if block.get("type") != _TEXT_BLOCK:
                continue  # image block; figures are found from their captions instead
            text_block = _to_text_block(block, page_number=index + 1, order=order)
            if text_block is not None:
                blocks.append(text_block)
                order += 1

        return RawPage(
            number=index + 1,
            width=float(page.rect.width),
            height=float(page.rect.height),
            blocks=tuple(blocks),
        )


def _to_text_block(block: dict[str, Any], *, page_number: int, order: int) -> TextBlock | None:
    """Flatten a PyMuPDF block into text plus its dominant style.

    A block's spans can mix sizes (a heading with a footnote marker, say). The dominant
    style is the one covering the most characters, which is what makes a heading look
    like a heading to the detector.
    """
    lines: list[str] = []
    weights: dict[tuple[float, str, bool, bool], int] = {}

    for line in block.get("lines", []):
        parts = []
        for span in line.get("spans", []):
            text = span.get("text", "")
            if not text:
                continue
            parts.append(text)
            flags = int(span.get("flags", 0))
            key = (
                round(float(span.get("size", 0.0)), 2),
                str(span.get("font", "")),
                bool(flags & _FLAG_BOLD),
                bool(flags & _FLAG_ITALIC),
            )
            weights[key] = weights.get(key, 0) + len(text)
        if parts:
            lines.append("".join(parts))

    text = "\n".join(lines).strip()
    if not text or not weights:
        return None

    size, font, bold, italic = max(weights.items(), key=lambda item: item[1])[0]
    if size <= 0:
        return None

    return TextBlock(
        text=text,
        page=page_number,
        index=order,
        bounding_box=BoundingBox.from_tuple(tuple(float(v) for v in block["bbox"])),  # type: ignore[arg-type]
        style=TextStyle(size=size, font=font, bold=bold, italic=italic),
    )


def _metadata_of(document: Any) -> PdfMetadata:
    raw = document.metadata or {}
    return PdfMetadata(
        title=_clean(raw.get("title")),
        author=_clean(raw.get("author")),
        subject=_clean(raw.get("subject")),
        keywords=_clean(raw.get("keywords")),
        creator=_clean(raw.get("creator")),
        producer=_clean(raw.get("producer")),
        creation_date=_clean(raw.get("creationDate")),
        modification_date=_clean(raw.get("modDate")),
    )


def _clean(value: str | None) -> str | None:
    if not value:
        return None
    collapsed = " ".join(value.split())
    return collapsed or None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()
