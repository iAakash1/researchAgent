#!/usr/bin/env python
"""Run discovery through evidence, then the existing agentic reasoning loop.

    uv run python scripts/run_full_research.py --goal "Why do multi-agent LLM systems fail?"

The API's /research/plan endpoint stops after evidence construction; the experiment
runner starts with an already-built corpus. This entry point connects those existing
runners for one real, inspectable research task without creating another pipeline.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from researchagent.container import build_container
from researchagent.schemas.workflow import ResearchConstraints


async def main(args: argparse.Namespace) -> int:
    container = build_container()
    try:
        print(f"Starting research: {args.goal}", flush=True)
        state = await container.workflow_runner.run(
            args.goal,
            constraints=ResearchConstraints(
                max_research_questions=args.max_questions,
                year_from=args.year_from,
            ),
        )
        if not state.succeeded:
            print(f"Pipeline stopped: {state.failure}", flush=True)
            return 1

        if args.build_graph:
            graph = await container.graph_builder.build(run_id=state.run_id)
            print(f"Graph: {graph.nodes} nodes, {graph.edges_accepted} edges", flush=True)

        print("Evidence ready; starting reasoning loop", flush=True)
        state = await container.reasoning_runner.run(state)
        session = state.reasoning
        if session is None or not session.terminated:
            print("Reasoning loop did not terminate cleanly", flush=True)
            return 1

        audits = await container.audit_trail.build(state)
        output = args.output or Path(f"evaluation/results/research_{state.run_id}.json")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                {
                    "run_id": state.run_id,
                    "goal": state.goal,
                    "status": state.status.value,
                    "plan": state.plan.model_dump(mode="json") if state.plan else None,
                    "discovery": (
                        state.discovery.model_dump(mode="json") if state.discovery else None
                    ),
                    "candidates": [c.model_dump(mode="json") for c in state.candidates],
                    "documents": (
                        state.documents.model_dump(mode="json") if state.documents else None
                    ),
                    "knowledge": (
                        state.knowledge.model_dump(mode="json") if state.knowledge else None
                    ),
                    "evidence": state.evidence.model_dump(mode="json") if state.evidence else None,
                    "history": [record.model_dump(mode="json") for record in state.history],
                    "reasoning": session.model_dump(mode="json"),
                    "termination_reason": (
                        session.termination_reason.value if session.termination_reason else None
                    ),
                    "accepted_findings": len(session.verified_findings),
                    "review_decision": (
                        session.latest_review.decision.value if session.latest_review else None
                    ),
                    "audits": [audit.model_dump(mode="json") for audit in audits],
                },
                indent=2,
            )
        )
        print(f"Wrote {output}", flush=True)
        if session.latest_review is None or not session.verified_findings:
            print(
                f"Research stopped without an accepted finding: {session.termination_reason}",
                flush=True,
            )
            return 1
        print(
            f"Completed {state.run_id}: {len(session.findings)} findings, "
            f"{len(session.verified_findings)} accepted",
            flush=True,
        )
        return 0
    finally:
        await container.aclose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goal", required=True)
    parser.add_argument("--max-questions", type=int, default=3)
    parser.add_argument("--year-from", type=int)
    parser.add_argument("--build-graph", action="store_true")
    parser.add_argument("--output", type=Path)
    raise SystemExit(asyncio.run(main(parser.parse_args())))
