#!/usr/bin/env python
"""Write metadata sidecars for the manual paper collection.

Scans ``storage/papers/raw/manual/`` and creates one JSON record per PDF under
``storage/papers/metadata/``. The PDFs themselves are read only — never moved, renamed
or modified.

Safe to re-run: existing records keep their pipeline flags and only their metadata is
refreshed.

    uv run python scripts/index_library.py
"""

from __future__ import annotations

import asyncio
import sys

from researchagent.container import build_container
from researchagent.models.library import PaperRecord
from researchagent.models.paper import SourceName


async def main() -> int:
    container = build_container()
    try:
        manual = next((s for s in container.paper_sources if s.name is SourceName.MANUAL), None)
        if manual is None:
            print("manual source is disabled in config/sources.yaml", file=sys.stderr)
            return 1

        health = await manual.health()
        if not health.healthy:
            print(f"manual library unavailable: {health.detail}", file=sys.stderr)
            return 1

        papers = manual.load_all()  # type: ignore[attr-defined]
        saved = await container.paper_repository.save_many(
            [
                PaperRecord(
                    paper=paper,
                    pdf_path=paper.local_path,
                    processing=PaperRecord(paper=paper).processing.mark(downloaded=True),
                )
                for paper in papers
            ]
        )

        print(f"indexed {len(saved)} papers -> {container.paper_repository.metadata_dir}")
        for record in saved:
            year = record.paper.year or "----"
            print(f"  {record.id:<14} {year}  {record.paper.title}")
        return 0
    finally:
        await container.aclose()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
