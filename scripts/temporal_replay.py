#!/usr/bin/env python
"""Temporal replay: add the papers published in (T0, T1] to a topic frozen at T0.

    uv run python scripts/temporal_replay.py --topic <run_id> --t0 2023 --t1 2025 \
        [--gold my_temporal_gold.json]

The topic must come from a research run whose corpus held only papers up to T0. Papers
in the window must already be ingested (parsed and extracted); they are passed to
maintenance as newly accepted. With ``--gold`` the diff is scored against your reviewed
annotations (template: evaluation/gold/temporal_replay.template.json, empty by design).
Runs maintenance, so it calls the configured LLM.
"""

from __future__ import annotations

import argparse
import asyncio
import pathlib

from researchagent.container import build_container
from researchagent.evaluation.temporal import (
    ReplayWindow,
    TemporalGoldSet,
    score_replay,
    split_corpus,
)
from researchagent.schemas.maintenance import MaintenanceRequest


async def main(args: argparse.Namespace) -> int:
    window = ReplayWindow(frozen_through_year=args.t0, added_through_year=args.t1)
    container = build_container()
    try:
        records = await container.paper_repository.list_all()
        split = split_corpus((record.paper for record in records), window)
        print(
            f"frozen {len(split.frozen)}, added {len(split.added)}, "
            f"after window {len(split.after_window)}, undated {len(split.undated)}"
        )
        diff = await container.maintenance.runner.run(
            args.topic, MaintenanceRequest(new_paper_ids=split.added)
        )
    finally:
        await container.aclose()
    out = pathlib.Path("evaluation/results") / f"temporal_{args.topic}_{args.t0}_{args.t1}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(diff.model_dump_json(indent=2))
    print(f"diff counts {diff.counts}; written {out}")
    if args.gold:
        score = score_replay(TemporalGoldSet.load(pathlib.Path(args.gold)), diff)
        print(score.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--topic", required=True)
    parser.add_argument("--t0", type=int, required=True)
    parser.add_argument("--t1", type=int, required=True)
    parser.add_argument("--gold")
    raise SystemExit(asyncio.run(main(parser.parse_args())))
