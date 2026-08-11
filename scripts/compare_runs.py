#!/usr/bin/env python
"""Compare two experiment artifacts.

    uv run python scripts/compare_runs.py run_a_ollama run_b_groq

Refuses to compare runs whose identities differ on anything but the model: two numbers
produced over different corpora or different retrieval configurations are not a
comparison, and printing them side by side would imply they were.
"""

from __future__ import annotations

import json
import pathlib
import sys

RESULTS_DIR = pathlib.Path("evaluation/experiments")
# Identity fields that must match. `llm_*` and `model_alias` are the variable under test.
INVARIANT = (
    "corpus_fingerprint",
    "papers",
    "knowledge_objects",
    "evidence_objects",
    "index_version",
    "embedding_model",
    "active_retriever",
    "config_digest",
)
ROWS = (
    ("findings_total", "findings"),
    ("verdict_verified", "verifier: VERIFIED"),
    ("accepted_findings", "reviewer: ACCEPTED"),
    ("partially_supported", "partially supported"),
    ("contradicted", "contradicted"),
    ("insufficient_evidence", "insufficient evidence"),
    ("unverifiable", "unverifiable"),
    ("cross_paper_findings", "cross-paper findings"),
    ("citation_completeness", "citation completeness"),
    ("evidence_coverage", "evidence coverage"),
    ("source_diversity", "source diversity"),
    ("unsupported_claim_rate", "unsupported claim rate"),
    ("review_decision", "review decision"),
    ("review_acceptance_rate", "review acceptance rate"),
    ("iterations", "iterations"),
    ("retrieval_attempts", "retrieval attempts"),
    ("tool_calls", "tool calls"),
    ("prompt_tokens", "input tokens"),
    ("completion_tokens", "output tokens"),
    ("total_tokens", "total tokens"),
    ("unmeasured_llm_calls", "unmeasured LLM calls"),
    ("graph_expansion_used", "graph expansion used"),
    ("termination_reason", "termination"),
    ("latency_ms", "latency (ms)"),
)


def main(labels: list[str]) -> int:
    runs = [json.loads((RESULTS_DIR / f"{label}.json").read_text()) for label in labels]

    mismatched = [field for field in INVARIANT if len({run["identity"][field] for run in runs}) > 1]
    if mismatched:
        print(f"NOT COMPARABLE — identities differ on: {mismatched}", file=sys.stderr)
        return 1

    print(
        f"corpus {runs[0]['identity']['corpus_fingerprint']} | "
        f"config {runs[0]['identity']['config_digest']} | "
        f"{runs[0]['identity']['papers']} papers, "
        f"{runs[0]['identity']['knowledge_objects']} objects"
    )
    print(f"question: {runs[0]['research_goal'][:96]}\n")

    # Wide enough for the longest cell as well as the longest label, so a long
    # termination reason cannot run into the next column.
    width = (
        max(
            *(len(label) for label in labels),
            *(len(str(run["metrics"][key])) for run in runs for key, _ in ROWS),
        )
        + 2
    )
    header = f"{'metric':<26}" + "".join(f"{label:>{width}}" for label in labels)
    print(header)
    print("-" * len(header))
    for key, name in ROWS:
        cells = "".join(f"{run['metrics'][key]!s:>{width + 8}}" for run in runs)
        print(f"{name:<26}{cells}")

    print("\nprovenance:")
    for label, run in zip(labels, runs, strict=True):
        untraceable = run["accepted_without_provenance"]
        verified = [f for f in run["findings"] if f["status"] == "verified"]
        print(f"  {label:<18}{len(verified)} accepted, {len(untraceable)} without a complete chain")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:] or ["run_a_ollama", "run_b_groq"]))
