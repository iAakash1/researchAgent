#!/usr/bin/env python
"""Run the agentic loop over the existing corpus.

    uv run python scripts/run_research.py "<research question>"

Skips discovery and extraction: the corpus is already built, so this exercises exactly the
v0.9 addition — retrieve, reason, verify, review — and prints the audit trail for every
finding it produces.
"""

from __future__ import annotations

import asyncio
import json
import pathlib
import sys
from datetime import UTC, datetime

from researchagent.container import build_container
from researchagent.models.research import (
    QuestionPriority,
    ResearchPlan,
    ResearchQuestion,
    SearchStrategy,
)
from researchagent.schemas.workflow import ResearchState

RESULTS_DIR = pathlib.Path("evaluation/results")


async def main(goal: str, questions: list[str]) -> int:
    container = build_container()
    try:
        plan = ResearchPlan(
            topic=goal[:60],
            framing=goal,
            research_questions=[
                ResearchQuestion(
                    id=f"RQ{index}",
                    question=text,
                    rationale="Supplied directly; this run exercises the reasoning loop.",
                    priority=QuestionPriority.HIGH,
                )
                for index, text in enumerate(questions, start=1)
            ],
            strategy=SearchStrategy(queries=[goal]),
        )
        state = ResearchState(goal=goal, plan=plan)

        final = await container.reasoning_runner.run(state)
        session = final.reasoning
        if session is None:
            print("no reasoning session was produced")
            return 1

        print(f"\niterations        {session.iteration}")
        print(f"bundles           {len(session.bundle_ids)}")
        print(f"tool calls        {len(session.tool_calls)}")
        print(f"hypotheses        {len(session.hypotheses)}")
        print(f"findings          {len(session.findings)}")
        print(f"verified          {len(session.verified_findings)}")
        print(
            f"termination       "
            f"{session.termination_reason.value if session.termination_reason else 'none'}"
        )

        review = session.latest_review
        if review is not None:
            print(f"\nreview            {review.decision.value}")
            print(f"  citation completeness {review.citation_completeness:.3f}")
            print(f"  source diversity      {review.source_diversity:.3f}")
            print(f"  unsupported rate      {review.unsupported_claim_rate:.3f}")
            for issue in review.blocking_issues[:6]:
                print(f"  BLOCKING {issue.finding_id}: {issue.code} — {issue.message}")

        audits = await container.audit_trail.build(final)
        for audit in audits:
            print(f"\n--- {audit.finding_id} [{audit.status.value}] {audit.question_id}")
            print(f'    "{audit.statement}"')
            for step in audit.steps:
                print(f"      {step.stage:<12} {step.actor:<12} {step.summary[:88]}")
            print(f"      provenance: {list(audit.provenance)[:4]}")
            print(f"      complete chain: {audit.is_complete}")

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        destination = RESULTS_DIR / (
            f"research_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json"
        )
        destination.write_text(
            json.dumps(
                {
                    "goal": goal,
                    "questions": questions,
                    "session": session.model_dump(mode="json"),
                    "audits": [audit.model_dump(mode="json") for audit in audits],
                },
                indent=2,
            )
        )
        print(f"\nwrote {destination}")
        return 0
    finally:
        await container.aclose()


if __name__ == "__main__":
    argv = sys.argv[1:]
    if not argv:
        print("usage: run_research.py '<goal>' ['<question>' ...]", file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(asyncio.run(main(argv[0], argv[1:] or [argv[0]])))
