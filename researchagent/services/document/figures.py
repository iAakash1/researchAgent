"""Figure and table detection.

Detected from their *captions*, not their pixels. A caption is text, is reliably
formatted ("Figure 3: ...", "Table II."), and carries the meaning a later stage actually
wants. Locating the image region itself would need layout analysis and would still tell
us nothing about what the figure shows.

No image understanding — that is explicitly out of scope, and would belong in a vision
model behind its own port.
"""

from __future__ import annotations

import re

from researchagent.core.logging import get_logger
from researchagent.models.document import Figure, Table
from researchagent.models.layout import RawDocument, TextBlock

logger = get_logger(__name__)

# The dash class spans hyphen, en dash and em dash: publishers use all three.
_FIGURE_CAPTION = re.compile(
    r"^\s*(?P<label>(?:fig(?:ure)?|abb)\.?\s*(?P<number>\d+[a-z]?|[IVXLC]+))\s*[.:—–-]?\s*"  # noqa: RUF001
    r"(?P<caption>.*)",
    re.IGNORECASE | re.DOTALL,
)
_TABLE_CAPTION = re.compile(
    r"^\s*(?P<label>tab(?:le)?\.?\s*(?P<number>\d+[a-z]?|[IVXLC]+))\s*[.:—–-]?\s*"  # noqa: RUF001
    r"(?P<caption>.*)",
    re.IGNORECASE | re.DOTALL,
)

# A caption block that is enormous is almost always body prose that happens to start
# with the word "Table"; a caption that is a bare label carries no information.
_MAX_CAPTION_CHARS = 1200


class FigureTableDetector:
    """Finds figure and table captions across the raw layout."""

    name = "figure_table_detector"

    def detect(self, document: RawDocument) -> tuple[tuple[Figure, ...], tuple[Table, ...]]:
        figures: list[Figure] = []
        tables: list[Table] = []

        for block in document.blocks:
            if block.is_blank or len(block.text) > _MAX_CAPTION_CHARS:
                continue

            text = " ".join(block.text.split())

            figure_match = _FIGURE_CAPTION.match(text)
            if figure_match and _is_caption(figure_match, block):
                figures.append(
                    Figure(
                        id=f"f{len(figures):03d}",
                        label=_normalise_label(figure_match.group("label")),
                        caption=figure_match.group("caption").strip(),
                        page=block.page,
                        bounding_box=block.bounding_box,
                    )
                )
                continue

            table_match = _TABLE_CAPTION.match(text)
            if table_match and _is_caption(table_match, block):
                tables.append(
                    Table(
                        id=f"t{len(tables):03d}",
                        label=_normalise_label(table_match.group("label")),
                        caption=table_match.group("caption").strip(),
                        page=block.page,
                        bounding_box=block.bounding_box,
                    )
                )

        logger.debug("figures_tables_detected", figures=len(figures), tables=len(tables))
        return tuple(figures), tuple(tables)


def _is_caption(match: re.Match[str], block: TextBlock) -> bool:
    """Guard against prose that merely begins with the word 'Table'.

    A real caption either has caption text after the label, or is a standalone label
    block. A sentence like "Table lookups dominate the cost" has neither a number nor a
    separator and is rejected by the pattern; this catches the remainder.
    """
    caption = match.group("caption").strip()
    if not caption:
        return True  # bare "Figure 3" label block, common when captions wrap
    # Captions start with a capital or a digit; mid-sentence continuations do not.
    return bool(caption[0].isupper() or caption[0].isdigit()) and len(caption) >= 3


def _normalise_label(label: str) -> str:
    collapsed = " ".join(label.split()).rstrip(".:")
    lowered = collapsed.lower()
    if lowered.startswith(("fig", "abb")):
        number = collapsed.split()[-1] if " " in collapsed else collapsed[3:].lstrip(".")
        return f"Figure {number}".strip()
    number = collapsed.split()[-1] if " " in collapsed else collapsed[3:].lstrip(".")
    return f"Table {number}".strip()
