#!/usr/bin/env python
"""One reproducible research experiment.

    uv run python scripts/run_experiment.py --alias reasoning        # local (Ollama)
    uv run python scripts/run_experiment.py --alias reasoning_remote # Groq GPT-OSS 120B

Runs the agentic loop over the existing corpus and writes a machine-readable artifact
containing the run's identity, its metrics, its findings and the provenance under each
one. Two artifacts are comparable only when their identities agree on everything except
the variable under test — which is why identity is recorded rather than assumed.

``--alias`` rebinds the reasoning, verification and reviewer agents only. Retrieval keeps
its configured model so the experimental variable is the reasoning engine, not the search.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import pathlib
import time
from datetime import UTC, datetime
from typing import Any

from researchagent.container import Container, build_container
from researchagent.models.reasoning import VerificationVerdict
from researchagent.models.research import (
    QuestionPriority,
    ResearchPlan,
    ResearchQuestion,
    SearchStrategy,
)
from researchagent.schemas.workflow import ResearchState

RESULTS_DIR = pathlib.Path("evaluation/experiments")
# Agents whose model is the experimental variable. Retrieval is held fixed.
REASONING_AGENTS = ("reasoning", "verification", "reviewer")


async def run_identity(container: Container, alias: str) -> dict[str, Any]:
    """Everything needed to decide whether two runs may be compared.

    Timestamps are recorded but are never the identity: two runs a minute apart over a
    changed corpus are different experiments, and two runs a week apart over the same one
    are not.
    """
    paper_ids = sorted(await container.knowledge_repository.list_ids())
    objects = 0
    evidence = 0
    for paper_id in paper_ids:
        knowledge = await container.knowledge_repository.get(paper_id)
        objects += len(knowledge.value.objects) if knowledge else 0
        stored = await container.evidence_repository.get_paper(paper_id)
        evidence += len(stored.records) if stored else 0

    versions = await container.graph_repository.versions()
    retrieval = container.retrieval_config
    spec = container.model_catalog.spec_for(alias)
    config_digest = hashlib.sha256(
        json.dumps(
            {
                "retrieval": retrieval.model_dump(mode="json"),
                "reasoning": container.reasoning_config.model_dump(mode="json"),
                "evidence": container.evidence_config.model_dump(mode="json"),
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()[:16]

    return {
        "corpus_fingerprint": hashlib.sha256(
            ("|".join(paper_ids) + f"#{objects}").encode()
        ).hexdigest()[:16],
        "papers": len(paper_ids),
        "knowledge_objects": objects,
        "evidence_objects": evidence,
        "graph_version": versions[0].identifier if versions else None,
        "index_version": f"{retrieval.embeddings.preprocessing_version}"
        f"-{retrieval.embeddings.model}",
        "embedding_model": retrieval.embeddings.model,
        "active_retriever": retrieval.active_retriever,
        "llm_provider": spec.provider,
        "llm_model": spec.model_name,
        "model_alias": alias,
        "config_digest": config_digest,
        "budget": container.reasoning_config.budget.model_dump(mode="json"),
    }


def metrics(session: Any, latency_ms: float) -> dict[str, Any]:
    """Measured outcomes only. Nothing here reads the prose."""
    verdicts = dict.fromkeys(VerificationVerdict, 0)
    for item in session.verifications:
        verdicts[item.verdict] += 1

    review = session.latest_review
    findings = session.findings
    cross_paper = sum(1 for finding in findings if finding.is_cross_paper)

    return {
        "findings_total": len(findings),
        # Two distinct things, deliberately reported separately: the verifier's opinion,
        # and what the reviewer actually accepted. A VERIFIED verdict is not acceptance —
        # a run that terminates before review has verdicts and no accepted findings.
        "verdict_verified": verdicts[VerificationVerdict.VERIFIED],
        "accepted_findings": len(session.verified_findings),
        "partially_supported": verdicts[VerificationVerdict.PARTIALLY_SUPPORTED],
        "contradicted": verdicts[VerificationVerdict.CONTRADICTED],
        "insufficient_evidence": verdicts[VerificationVerdict.INSUFFICIENT_EVIDENCE],
        "unverifiable": verdicts[VerificationVerdict.UNVERIFIABLE],
        "hypotheses": len(session.hypotheses),
        "cross_paper_findings": cross_paper,
        "citation_completeness": review.citation_completeness if review else None,
        "evidence_coverage": review.evidence_coverage if review else None,
        "source_diversity": review.source_diversity if review else None,
        "unsupported_claim_rate": review.unsupported_claim_rate if review else None,
        "review_decision": review.decision.value if review else None,
        "review_accepted": len(review.accepted_findings) if review else 0,
        "review_rejected": len(review.rejected_findings) if review else 0,
        "review_acceptance_rate": (
            round(len(review.accepted_findings) / len(findings), 4) if review and findings else 0.0
        ),
        "iterations": session.iteration,
        "retrieval_attempts": session.ledger.retrieval_attempts,
        "tool_calls": session.ledger.tool_calls,
        "bundles": len(session.bundle_ids),
        "prompt_tokens": session.ledger.prompt_tokens,
        "completion_tokens": session.ledger.completion_tokens,
        "total_tokens": session.ledger.total_tokens,
        "tokens_by_agent": dict(session.ledger.tokens_by_agent),
        "unmeasured_llm_calls": session.ledger.unmeasured_calls,
        "termination_reason": (
            session.termination_reason.value if session.termination_reason else None
        ),
        "latency_ms": round(latency_ms, 1),
        "graph_expansion_used": any(
            call.tool.value == "search_graph" for call in session.tool_calls
        ),
    }


async def main(args: argparse.Namespace) -> int:
    container = build_container()
    try:
        if args.alias != "reasoning":
            # Rebind only the reasoning-side agents; retrieval stays on its own model so
            # the two runs search identically.
            container.agent_config.agents.update(
                {
                    name: container.agent_config.spec_for(name).model_copy(
                        update={"model": args.alias}
                    )
                    for name in REASONING_AGENTS
                }
            )

        if args.build_graph:
            # The default graph backend is in-memory, so a generation does not survive
            # the process that built it. Building here is what makes graph expansion
            # actually available to the retrieval agent rather than nominally available.
            report = await container.graph_builder.build(run_id="experiment")
            print(f"graph: {report.nodes} nodes, {report.edges_accepted} edges")

        identity = await run_identity(container, args.alias)
        plan = ResearchPlan(
            topic=args.goal[:60],
            framing=args.goal,
            research_questions=[
                ResearchQuestion(
                    id=f"RQ{index}",
                    question=text,
                    rationale="Supplied by the experiment; this run exercises the loop.",
                    priority=QuestionPriority.HIGH,
                )
                for index, text in enumerate(args.questions, start=1)
            ],
            strategy=SearchStrategy(queries=[args.goal]),
        )
        state = ResearchState(goal=args.goal, plan=plan)

        started = time.perf_counter()
        final = await container.reasoning_runner.run(state)
        latency_ms = (time.perf_counter() - started) * 1000
        session = final.reasoning
        if session is None:
            print("no reasoning session produced")
            return 1

        audits = await container.audit_trail.build(final)
        measured = metrics(session, latency_ms)

        accepted = set(session.latest_review.accepted_findings if session.latest_review else ())
        untraceable = [
            audit.finding_id
            for audit in audits
            if audit.finding_id in accepted and not audit.is_complete
        ]

        artifact = {
            "run_id": final.run_id,
            "label": args.label,
            "identity": identity,
            "research_goal": args.goal,
            "research_questions": [q.question for q in plan.research_questions],
            "metrics": measured,
            "findings": [
                {
                    "id": finding.id,
                    "question_id": finding.question_id,
                    "statement": finding.statement,
                    "status": finding.status.value,
                    "papers": list(finding.paper_ids),
                    "bundle_ids": list(finding.bundle_ids),
                    "verdict": (
                        v.verdict.value
                        if (v := session.verification_for(finding.id)) is not None
                        else None
                    ),
                    "overstatements": list(v.overstatements) if v else [],
                }
                for finding in session.findings
            ],
            "audits": [audit.model_dump(mode="json") for audit in audits],
            "accepted_without_provenance": untraceable,
            "run_at": datetime.now(UTC).isoformat(),
        }

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        destination = RESULTS_DIR / f"{args.label}.json"
        destination.write_text(json.dumps(artifact, indent=2))

        print(f"\n[{args.label}] {identity['llm_provider']}:{identity['llm_model']}")
        for key in (
            "findings_total",
            "verdict_verified",
            "accepted_findings",
            "partially_supported",
            "contradicted",
            "insufficient_evidence",
            "unverifiable",
            "citation_completeness",
            "source_diversity",
            "unsupported_claim_rate",
            "review_decision",
            "iterations",
            "retrieval_attempts",
            "tool_calls",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "unmeasured_llm_calls",
            "graph_expansion_used",
            "termination_reason",
            "latency_ms",
        ):
            print(f"  {key:<26}{measured[key]}")
        if untraceable:
            print(f"  ACCEPTED WITHOUT PROVENANCE: {untraceable}")
        print(f"wrote {destination}")
        return 1 if untraceable else 0
    finally:
        await container.aclose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--alias", default="reasoning")
    parser.add_argument("--label", required=True)
    parser.add_argument("--goal", required=True)
    parser.add_argument("--question", dest="questions", action="append", required=True)
    parser.add_argument("--build-graph", action="store_true")
    raise SystemExit(asyncio.run(main(parser.parse_args())))
