#!/usr/bin/env python
"""Resolve hand-written relevance judgements into a versioned gold set.

The judgements below were written by reading the extracted knowledge corpus and deciding,
per query, which objects answer it. No model was asked to label anything: this script only
resolves the chosen names to the ids they currently have.

Every entry is written as `draft`. A human must read the resolved objects and flip the
status to `reviewed` before the benchmark treats the numbers as real.

    uv run python scripts/build_gold_set.py
"""

from __future__ import annotations

import asyncio
import json
import pathlib
import sys

from researchagent.evaluation.gold import (
    GoldJudgement,
    GoldQuery,
    GoldSet,
    Relevance,
    ReviewStatus,
)
from researchagent.models.knowledge import KnowledgeKind

HIGH, MED, LOW = Relevance.HIGHLY_RELEVANT, Relevance.RELEVANT, Relevance.MARGINAL

# query id -> (question, kind filter, {object name: (grade, rationale)})
JUDGEMENTS: dict[str, tuple[str, tuple[KnowledgeKind, ...], dict[str, tuple[Relevance, str]]]] = {
    "GQ1": (
        "What triggers metastable failures in distributed systems?",
        (),
        {
            "Trigger vs. Root Cause": (HIGH, "Directly distinguishes trigger from root cause"),
            "trigger_type_percentage on Table 1": (HIGH, "Measured distribution of trigger types"),
            "load_spike_trigger_percentage on Table 1": (
                HIGH,
                "Load spikes as a trigger, measured",
            ),
            "Change of Policy during Overload": (MED, "Policy change is a named trigger path"),
            "retry case study": (MED, "Retries are a documented trigger"),
            "load shedding": (LOW, "Mitigation of the overload that triggers failure"),
        },
    ),
    "GQ2": (
        "Which techniques mitigate overload in distributed systems?",
        (KnowledgeKind.METHOD,),
        {
            "Circuit Breaker Pattern": (HIGH, "Canonical overload mitigation"),
            "Prioritization": (HIGH, "Prioritising work under overload"),
            "Adaptive Policies": (HIGH, "Policies that adapt to load"),
            "Autoscaling": (MED, "Capacity response to load"),
            "Lower Priority for Retried Queries": (MED, "Retry-specific mitigation"),
            "Fast Error Paths": (LOW, "Reduces cost of failures under load"),
        },
    ),
    "GQ3": (
        "What metrics measure system performance under load?",
        (KnowledgeKind.METRIC,),
        {
            "goodput": (HIGH, "The primary throughput metric in this literature"),
            "latency": (HIGH, "Primary latency metric"),
            "Queue Length": (MED, "Queue depth as a load indicator"),
            "GC Duration": (MED, "Pause time under load"),
            "PrintGCApplication-StoppedTime": (LOW, "A specific GC measurement"),
        },
    ),
    "GQ4": (
        "What limitations do the authors acknowledge in modelling metastable failures?",
        (KnowledgeKind.LIMITATION,),
        {
            "Simplified models may not capture metastability": (HIGH, "Explicit modelling limit"),
            "Perfect fidelity in models is not achievable": (HIGH, "Explicit modelling limit"),
            "Precise simulation of complex systems is computationally expensive": (
                MED,
                "Cost limit on simulation",
            ),
            "Limited information in reports": (LOW, "Data limitation, not a modelling one"),
        },
    ),
    "GQ5": (
        "What future research directions are proposed for metastable failures?",
        (KnowledgeKind.FUTURE_WORK,),
        {
            "Designing systems to avoid metastable failures": (HIGH, "Named future direction"),
            "Accurately modeling and reproducing metastable failures": (HIGH, "Named direction"),
            "Estimating the probability of novel metastable failures": (HIGH, "Named direction"),
            "Identifying vulnerabilities in existing systems": (MED, "Named direction"),
            "Develop solutions to metastable failures.": (MED, "Named direction, second paper"),
            "Understanding the impact of system changes on sustaining effects": (
                MED,
                "Named direction",
            ),
        },
    ),
    "GQ6": (
        "How do caches contribute to metastable failure?",
        (),
        {
            "Read-Through Cache": (HIGH, "Cache design implicated in the failure mode"),
            "read-through cache": (HIGH, "Same concept, extracted separately"),
            "look-aside cache": (HIGH, "The look-aside cache case study"),
        },
    ),
}

GOLD_PATH = pathlib.Path("evaluation/gold/retrieval_v1.json")


async def main() -> int:
    knowledge_dir = pathlib.Path("storage/papers/knowledge")
    if not knowledge_dir.is_dir():  # noqa: ASYNC240 - CLI entry point, no event loop concerns
        print("no knowledge corpus on disk; run the v0.5 pipeline first", file=sys.stderr)
        return 1

    by_name: dict[str, list[tuple[str, str, list[str]]]] = {}
    papers: list[str] = []
    for path in sorted(knowledge_dir.glob("*.json")):  # noqa: ASYNC240 - CLI entry point
        stored = json.loads(path.read_text())["value"]
        papers.append(stored["paper_id"])
        for obj in stored["objects"]:
            by_name.setdefault(obj["name"], []).append(
                (obj["id"], obj["kind"], [e["id"] for e in obj["evidence"]])
            )

    queries = []
    for query_id, (text, kinds, wanted) in JUDGEMENTS.items():
        judgements, evidence_ids, missing = [], [], []
        for name, (grade, rationale) in wanted.items():
            matches = by_name.get(name)
            if not matches:
                missing.append(name)
                continue
            for object_id, _, evidence in matches:
                judgements.append(
                    GoldJudgement(
                        knowledge_object_id=object_id, relevance=grade, rationale=rationale
                    )
                )
                evidence_ids.extend(evidence)

        if missing:
            print(f"  {query_id}: {len(missing)} name(s) not in corpus: {missing}")

        queries.append(
            GoldQuery(
                id=query_id,
                text=text,
                kinds=kinds,
                judgements=tuple(judgements),
                relevant_evidence_ids=tuple(dict.fromkeys(evidence_ids)),
                status=ReviewStatus.DRAFT,
                notes="Judgements written by inspecting the corpus; awaiting human review.",
            )
        )

    gold = GoldSet(
        version="v1",
        corpus_description=f"{len(papers)} manually collected papers: {', '.join(papers)}",
        queries=tuple(queries),
    )
    gold.save(GOLD_PATH)

    print(f"\nwrote {GOLD_PATH} — {len(gold.queries)} queries, all DRAFT")
    for query in gold.queries:
        print(
            f"  {query.id}: {len(query.judgements):>2} judged objects, "
            f"{len(query.relevant_evidence_ids):>2} evidence  | {query.text[:52]}"
        )
    print("\nReview the judgements, then set status to 'reviewed' before trusting the numbers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
