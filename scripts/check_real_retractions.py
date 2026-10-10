#!/usr/bin/env python
"""Check a user-supplied list of real DOIs against the integrity registry (Crossref).

    cp evaluation/gold/real_retractions.template.json my_retractions.json   # then fill it
    uv run python scripts/check_real_retractions.py my_retractions.json

Contacts the live Crossref REST API (free; set contact_email in config/sources.yaml for
the polite pool). Reports how often the registry agrees with the statuses you supplied;
failed lookups are counted separately and never as agreement.
"""

from __future__ import annotations

import asyncio
import pathlib
import sys

from researchagent.container import build_container
from researchagent.evaluation.real_retractions import RealRetractionSet, check_against_registry


async def main(path: str) -> int:
    dataset = RealRetractionSet.load(pathlib.Path(path))
    if not dataset.entries:
        print("the file has no entries; fill `entries` first (nothing is ever pre-filled)")
        return 2
    container = build_container()
    try:
        report = await check_against_registry(dataset, container.maintenance.integrity)
    finally:
        await container.aclose()
    for row in report.rows:
        verdict = {True: "agrees", False: "DISAGREES", None: "lookup failed"}[row.agrees]
        print(
            f"{row.doi:<40} expected {row.expected.value:<22} "
            f"registry {row.observed.value:<22} {verdict}"
        )
    print(
        f"\n{report.agreed} agree, {report.disagreed} disagree, "
        f"{report.lookup_failures} lookup failures; agreement rate {report.agreement_rate}"
    )
    out = pathlib.Path("evaluation/results") / f"real_retractions_{dataset.version}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(asyncio.run(main(sys.argv[1])))
