#!/usr/bin/env python
"""Build the retrieval gold set from the validated corpus.

    uv run python scripts/build_gold_set.py [--out PATH]

Ground truth is derived by *reading the corpus* — the judgements below were written by
inspecting the extracted KnowledgeObjects and deciding, per query, which ones answer it.
No model was asked to produce a relevance label: a benchmark whose labels came from a
language model measures agreement with that model, not retrieval quality.

Every object id is verified against the repository before the file is written, so a query
referring to something extraction no longer produces fails loudly instead of quietly
scoring zero.

Everything is emitted as `draft`. Promoting a query to `reviewed` is a human act: read the
objects, confirm or amend the judgements, then set the status and `reviewed_by`. Only
reviewed queries may back a claim about retrieval quality.
"""

from __future__ import annotations

import asyncio
import pathlib
import sys

from researchagent.container import build_container
from researchagent.evaluation import GoldJudgement, GoldQuery, GoldSet, Relevance
from researchagent.models.knowledge import KnowledgeKind

GOLD_PATH = pathlib.Path("evaluation/gold/retrieval_v1.json")

HIGH, REL, MARG = Relevance.HIGHLY_RELEVANT, Relevance.RELEVANT, Relevance.MARGINAL

# (query id, text, kinds, [(object id, relevance, rationale)], intent note)
QUERIES: list[tuple[str, str, tuple[KnowledgeKind, ...], list[tuple[str, Relevance, str]], str]] = [
    # ---- method intent -----------------------------------------------------
    (
        "GQ1",
        "What techniques mitigate overload in distributed systems?",
        (KnowledgeKind.METHOD,),
        [
            (
                "manual:01#method:prioritization:1",
                HIGH,
                "Priorities retain efficiency under exhaustion",
            ),
            (
                "manual:01#method:circuit-breaker-pattern:8",
                HIGH,
                "Blocks requests to break the loop",
            ),
            (
                "manual:01#method:autoscaling:6",
                HIGH,
                "Adds capacity to escape the metastable state",
            ),
            (
                "manual:01#method:change-of-policy-during-overload:0",
                HIGH,
                "Directly named overload policy",
            ),
            (
                "manual:01#method:lower-priority-for-retried-queries:12",
                REL,
                "Stops retries perpetuating overload",
            ),
            ("manual:01#method:server-policy:11", REL, "Switches to a goodput-maximising policy"),
            ("manual:01#method:adaptive-policies:9", REL, "Adaptive retry/failover decisions"),
            (
                "manual:01#method:read-through-cache:13",
                MARG,
                "Raises hit rate, indirectly reduces load",
            ),
        ],
        "method",
    ),
    (
        "GQ2",
        "How does TCP avoid congestion collapse?",
        (KnowledgeKind.METHOD,),
        [
            ("manual:03#method:slow-start:5", HIGH, "The core congestion-avoidance algorithm"),
            (
                "manual:03#method:packet-conservation-principle:1",
                HIGH,
                "The stability principle behind it",
            ),
            ("manual:03#method:acks-as-clock:4", HIGH, "Self-clocking is the mechanism"),
            ("manual:03#method:conservation-of-packets:2", REL, "Restatement of the principle"),
            ("manual:03#method:slow-start-window-increase:7", REL, "The window increase rule"),
            (
                "manual:03#method:window-based-transport-protocol:0",
                MARG,
                "The protocol family, not the fix",
            ),
        ],
        "method",
    ),
    (
        "GQ3",
        "What guard mechanisms prevent error cascades between agents?",
        (KnowledgeKind.METHOD,),
        [
            ("manual:11#method:control-flow-guard:6", HIGH, "Named guard against cascades"),
            ("manual:11#method:governance-layer:1", HIGH, "Governance constrains agent actions"),
            ("manual:11#method:middleware-module:0", REL, "Interception point for the guard"),
            ("manual:12#method:acp:0", REL, "Agent Control Protocol governs agent behaviour"),
            (
                "manual:10#method:self-verification-step:4",
                MARG,
                "Verification catches errors before spread",
            ),
        ],
        "method",
    ),
    # ---- dataset intent ----------------------------------------------------
    (
        "GQ4",
        "Which benchmark datasets are used to evaluate multi-agent LLM systems?",
        (KnowledgeKind.DATASET,),
        [
            ("manual:10#dataset:mast-data:0", HIGH, "The paper's own failure-taxonomy dataset"),
            ("manual:10#dataset:mast-data-human:1", HIGH, "Human-annotated variant"),
            ("manual:10#dataset:programdev-v2:4", REL, "Task suite the systems are run on"),
            ("manual:10#dataset:olympiadbench:7", REL, "Reasoning benchmark used"),
            ("manual:07#dataset:ms-marco:0", REL, "Retrieval benchmark in ProtocolBench"),
            ("manual:07#dataset:2wikimulti:1", REL, "Multi-hop QA benchmark"),
            (
                "manual:06#dataset:ai-nativebench:0",
                MARG,
                "Agent benchmark, not multi-agent specifically",
            ),
        ],
        "dataset",
    ),
    (
        "GQ5",
        "Which multi-agent frameworks were studied?",
        (KnowledgeKind.DATASET,),
        [
            ("manual:10#dataset:metagpt:2", HIGH, "Named framework under study"),
            ("manual:10#dataset:chatdev:3", HIGH, "Named framework under study"),
            ("manual:10#dataset:ag2-(mathchat):8", HIGH, "Named framework under study"),
        ],
        "dataset",
    ),
    (
        "GQ6",
        "What example applications reproduce metastable failures?",
        (KnowledgeKind.DATASET, KnowledgeKind.METHOD),
        [
            (
                "manual:02#method:example-applications:2",
                HIGH,
                "Explicitly a set of reproducing applications",
            ),
            ("manual:02#dataset:java-program:0", HIGH, "The multi-threaded reproducer"),
            (
                "manual:02#dataset:rps-(requests-per-second):1",
                REL,
                "The load knob used to induce failure",
            ),
            ("manual:01#dataset:retry-case-study:8", REL, "Case study of a retry-driven failure"),
            ("manual:01#dataset:kraken:13", MARG, "Named system in the discussion"),
        ],
        "dataset",
    ),
    # ---- metric intent -----------------------------------------------------
    (
        "GQ7",
        "Which metrics indicate a system is in a metastable state?",
        (KnowledgeKind.METRIC,),
        [
            ("manual:01#metric:goodput:0", HIGH, "Throughput of useful work — the defining signal"),
            ("manual:01#metric:latency:1", HIGH, "Latency is the other characteristic metric"),
            ("manual:02#metric:queue-length:0", HIGH, "Queue growth marks the sustaining loop"),
            ("manual:02#metric:gc-duration:1", REL, "GC pauses amplify the loop"),
            (
                "manual:02#metric:queue-length-vs.-gc-duration-correlation:2",
                REL,
                "The correlation is the evidence",
            ),
            (
                "manual:01#method:minimum-queueing-latency:10",
                MARG,
                "Uses queueing latency as the signal",
            ),
        ],
        "metric",
    ),
    (
        "GQ8",
        "What client-side metrics does gRPC expose for retries?",
        (KnowledgeKind.METRIC,),
        [
            ("manual:15#metric:grpc-client-attempt-started:0", HIGH, "Counts retry attempts"),
            ("manual:15#metric:grpc-client-attempt-duration:1", HIGH, "Per-attempt duration"),
            (
                "manual:15#metric:grpc-client-call-duration:4",
                HIGH,
                "Whole-call duration across retries",
            ),
            (
                "manual:15#metric:grpc-client-sent-total-compressed-message-size:2",
                REL,
                "Client-side volume",
            ),
            (
                "manual:15#metric:grpc-client-received-total-compressed-message-si:3",
                REL,
                "Client-side volume",
            ),
        ],
        "metric",
    ),
    (
        "GQ9",
        "How is agent naming and directory performance measured?",
        (KnowledgeKind.METRIC,),
        [
            ("manual:13#metric:latency:4", HIGH, "Lookup latency"),
            ("manual:13#metric:write-overhead:0", HIGH, "Cost of updates to the index"),
            ("manual:13#metric:write-frequency:1", HIGH, "How often the index changes"),
            ("manual:13#metric:fault-tolerance:5", REL, "Resilience of the directory"),
            (
                "manual:13#metric:governance-complexity:2",
                MARG,
                "Governance rather than performance",
            ),
        ],
        "metric",
    ),
    # ---- result intent -----------------------------------------------------
    (
        "GQ10",
        "What proportion of metastable failures are caused by load spikes?",
        (KnowledgeKind.RESULT,),
        [
            (
                "manual:02#result:load_spike_trigger_percentage-on-table-1:1",
                HIGH,
                "The measured share, directly",
            ),
            (
                "manual:02#result:trigger_type_percentage-on-table-1:0",
                HIGH,
                "The trigger distribution it sits in",
            ),
        ],
        "result",
    ),
    (
        "GQ11",
        "What failure rates were measured for multi-agent frameworks?",
        (KnowledgeKind.RESULT,),
        [
            (
                "manual:10#result:failure-rate-on-metagpt-and-chatdev-frameworks:0",
                HIGH,
                "The measured rate",
            ),
            ("manual:10#metric:failure-rate:0", REL, "The metric being reported"),
        ],
        "result",
    ),
    (
        "GQ12",
        "What throughput did the protocol sustain under attack?",
        (KnowledgeKind.RESULT,),
        [
            (
                "manual:12#result:throughput-on-experiment-1:-cooldown-evasion-att:2",
                HIGH,
                "Throughput under the attack",
            ),
            (
                "manual:12#result:requests_processed_before_first_block-on-experim:4",
                HIGH,
                "Requests before first block",
            ),
            ("manual:12#dataset:cooldown-evasion-attack:3", REL, "The attack scenario measured"),
        ],
        "result",
    ),
    (
        "GQ13",
        "How much bandwidth was recovered by the congestion fix?",
        (KnowledgeKind.RESULT,),
        [
            ("manual:03#result:bandwidth-utilization:5", HIGH, "The measured utilisation"),
            ("manual:03#result:retransmission-rate:6", HIGH, "Retransmissions before/after"),
            ("manual:03#metric:bandwidth-utilization:0", REL, "The metric definition"),
            ("manual:03#result:window-open-time:3", MARG, "Related timing measurement"),
        ],
        "result",
    ),
    # ---- limitation intent -------------------------------------------------
    (
        "GQ14",
        "What are the limitations of modelling and simulating metastable failures?",
        (KnowledgeKind.LIMITATION,),
        [
            (
                "manual:01#limitation:simplified-models-may-not-capture-metastability:0",
                HIGH,
                "Abstraction hides the effect",
            ),
            (
                "manual:01#limitation:perfect-fidelity-in-models-is-not-achievable:1",
                HIGH,
                "Fidelity ceiling",
            ),
            (
                "manual:01#limitation:precise-simulation-of-complex-systems-is-computa:2",
                HIGH,
                "Simulation cost",
            ),
            (
                "manual:01#limitation:reconfiguration-as-a-recovery-strategy-has-limit:3",
                REL,
                "Recovery strategy limits",
            ),
            (
                "manual:02#limitation:limited-information-in-reports:1",
                REL,
                "Incident reports are incomplete",
            ),
        ],
        "limitation",
    ),
    (
        "GQ15",
        "What limitations do LLM benchmarks acknowledge?",
        (KnowledgeKind.LIMITATION,),
        [
            (
                "manual:06#limitation:non-determinism-of-llms:0",
                HIGH,
                "Non-determinism undermines repeatability",
            ),
            (
                "manual:06#limitation:opacity-of-model-updates:1",
                HIGH,
                "Models change under the benchmark",
            ),
            ("manual:06#limitation:prompt-sensitivity:4", HIGH, "Results move with the prompt"),
            (
                "manual:06#limitation:metric-bias-in-correctness-measurement:3",
                HIGH,
                "The metric itself is biased",
            ),
            (
                "manual:06#limitation:generalizability-to-future-models-and-domains:2",
                REL,
                "External validity",
            ),
            ("manual:07#limitation:model-fixed-in-experiments:2", REL, "Single model tested"),
            ("manual:07#limitation:limited-scenarios:0", REL, "Scenario coverage"),
        ],
        "limitation",
    ),
    (
        "GQ16",
        "What security risks are identified in agent protocol specifications?",
        (KnowledgeKind.LIMITATION,),
        [
            ("manual:05b#limitation:arbitrary-code-execution-risk:4", HIGH, "Code execution risk"),
            ("manual:05b#limitation:untrusted-tool-behavior:5", HIGH, "Tools cannot be trusted"),
            (
                "manual:05b#limitation:no-unauthorized-data-transmission:2",
                HIGH,
                "Data exfiltration constraint",
            ),
            (
                "manual:05b#limitation:security-enforcement:0",
                REL,
                "Enforcement is left to implementors",
            ),
            (
                "manual:05b#limitation:user-consent-for-tool-invocation:6",
                REL,
                "Consent requirement",
            ),
            ("manual:05b#limitation:data-protection-requirements:3", REL, "Data protection"),
        ],
        "limitation",
    ),
    # ---- future work intent ------------------------------------------------
    (
        "GQ17",
        "What future work is proposed on metastable failures?",
        (KnowledgeKind.FUTURE_WORK,),
        [
            (
                "manual:01#future_work:designing-systems-to-avoid-metastable-failures:0",
                HIGH,
                "Frameworks to avoid them",
            ),
            (
                "manual:01#future_work:estimating-the-probability-of-novel-metastable-f:1",
                HIGH,
                "Finding unknown failures",
            ),
            (
                "manual:01#future_work:accurately-modeling-and-reproducing-metastable-f:4",
                HIGH,
                "Modelling and reproduction",
            ),
            (
                "manual:01#future_work:identifying-vulnerabilities-in-existing-systems:2",
                REL,
                "Vulnerability discovery",
            ),
            (
                "manual:01#future_work:resolving-metastable-failures-with-elastic-cloud:3",
                REL,
                "Elastic capacity",
            ),
            (
                "manual:02#future_work:develop-solutions-to-metastable-failures.:1",
                REL,
                "Calls for solutions",
            ),
            (
                "manual:02#future_work:encourage-further-research-on-metastable-failure:0",
                MARG,
                "A call for research",
            ),
        ],
        "future_work",
    ),
    (
        "GQ18",
        "What future directions are proposed for agent governance?",
        (KnowledgeKind.FUTURE_WORK,),
        [
            (
                "manual:12#future_work:make-governance-detectable,-measurable,-and-form:2",
                HIGH,
                "Formalising governance",
            ),
            (
                "manual:12#future_work:mitigate-cross-agent-coordination-attacks:0",
                HIGH,
                "Coordination attacks",
            ),
            (
                "manual:11#future_work:develop-more-governance-mechanisms:2",
                HIGH,
                "More governance mechanisms",
            ),
            ("manual:13#future_work:global-governance:1", REL, "Governance at index scale"),
            (
                "manual:12#future_work:improve-acp's-efficiency-and-scalability:1",
                MARG,
                "Efficiency, not governance",
            ),
        ],
        "future_work",
    ),
    # ---- cross-paper comparison -------------------------------------------
    (
        "GQ19",
        "How do different papers characterise the triggers of cascading failure?",
        (),
        [
            (
                "manual:01#method:trigger-vs.-root-cause:7",
                HIGH,
                "Distinguishes trigger from root cause",
            ),
            (
                "manual:02#result:trigger_type_percentage-on-table-1:0",
                HIGH,
                "Measured trigger distribution",
            ),
            ("manual:02#method:identification-and-classification:1", HIGH, "Taxonomy of triggers"),
            (
                "manual:03#method:packet-conservation-failure:3",
                REL,
                "Failure mode in a different domain",
            ),
            ("manual:11#method:control-flow-guard:6", REL, "Cascade prevention in agent systems"),
            (
                "manual:02#result:load_spike_trigger_percentage-on-table-1:1",
                REL,
                "One trigger, quantified",
            ),
        ],
        "cross_paper",
    ),
    (
        "GQ20",
        "Which papers evaluate retry behaviour and what do they conclude?",
        (),
        [
            (
                "manual:01#method:lower-priority-for-retried-queries:12",
                HIGH,
                "Retry policy recommendation",
            ),
            ("manual:01#dataset:retry-case-study:8", HIGH, "The retry case study"),
            (
                "manual:15#limitation:no-default-retry-policy:0",
                HIGH,
                "gRPC has no default retry policy",
            ),
            ("manual:15#limitation:disable-retries:1", REL, "Retries can be disabled"),
            ("manual:01#method:adaptive-policies:9", REL, "Adaptive retry decisions"),
            ("manual:15#metric:grpc-client-attempt-started:0", REL, "Retry attempts are measured"),
        ],
        "cross_paper",
    ),
    (
        "GQ21",
        "What evaluation harnesses do agent benchmark papers build?",
        (),
        [
            ("manual:07#method:protocolbench:0", HIGH, "The harness itself"),
            ("manual:07#method:scenario-harness:3", HIGH, "Named scenario harness"),
            ("manual:06#method:ai-nativebench:1", HIGH, "The benchmark system"),
            (
                "manual:06#method:trace-first-evaluation-methodology:0",
                HIGH,
                "Its evaluation methodology",
            ),
            ("manual:07#method:protocolrouterbench:5", REL, "Router-specific harness"),
            ("manual:10#method:mast-development-process:2", REL, "Taxonomy construction process"),
            ("manual:07#method:logging-&-metrics-stack:4", MARG, "Supporting infrastructure"),
        ],
        "cross_paper",
    ),
    # ---- method -> dataset relationship ------------------------------------
    (
        "GQ22",
        "Which datasets was AI-NativeBench evaluated on?",
        (),
        [
            ("manual:06#dataset:markdown-validator:1", HIGH, "One of the two task datasets"),
            ("manual:06#dataset:landing-page-generator:2", HIGH, "The other task dataset"),
            ("manual:06#dataset:ai-nativebench:0", REL, "The suite these sit in"),
            ("manual:06#method:ai-nativebench:1", REL, "The method being evaluated"),
        ],
        "method_dataset",
    ),
    (
        "GQ23",
        "Which datasets does the ACP reference implementation exercise?",
        (),
        [
            (
                "manual:12#dataset:go-reference-implementation:0",
                HIGH,
                "The implementation under test",
            ),
            ("manual:12#dataset:multi-org-demo:1", HIGH, "Deployment scenario"),
            ("manual:12#dataset:payment-agent:2", HIGH, "Agent scenario"),
            ("manual:12#dataset:cooldown-evasion-attack:3", REL, "Attack scenario exercised"),
            (
                "manual:12#dataset:distributed-multi-agent-attack:4",
                REL,
                "Attack scenario exercised",
            ),
            ("manual:12#method:acp:0", REL, "The protocol being exercised"),
        ],
        "method_dataset",
    ),
    # ---- method -> result relationship -------------------------------------
    (
        "GQ24",
        "What results did ProtocolBench report for latency?",
        (),
        [
            (
                "manual:07#result:per_request_latency-on-cross-model-streaming-que:2",
                HIGH,
                "The measured latency",
            ),
            ("manual:07#metric:p99-latency:0", HIGH, "The latency metric used"),
            ("manual:07#metric:latency-ranking:1", REL, "Comparative ranking"),
            ("manual:07#method:protocolbench:0", REL, "The system producing the result"),
        ],
        "method_result",
    ),
    (
        "GQ25",
        "What did the NANDA index measure about DNS?",
        (),
        [
            ("manual:13#result:daily-lookups-on-dns-infrastructure:0", HIGH, "DNS lookup volume"),
            ("manual:13#result:write-frequency-on-dns:1", HIGH, "DNS write frequency"),
            ("manual:13#metric:write-frequency:1", REL, "The metric definition"),
            ("manual:13#method:nanda-index:6", REL, "The system making the comparison"),
        ],
        "method_result",
    ),
    (
        "GQ26",
        "What costs does self-healing introduce in agent systems?",
        (),
        [
            (
                "manual:06#result:cost-multiplier-effect-of-self-healing-mechanism:2",
                HIGH,
                "The measured cost multiplier",
            ),
            ("manual:06#result:inference-dominance:1", HIGH, "Inference dominates cost"),
            ("manual:06#metric:cost-performance-trade-off:2", REL, "The trade-off metric"),
            ("manual:11#limitation:latency-overhead:3", REL, "Guard overhead in a related system"),
        ],
        "method_result",
    ),
]


async def main(out: pathlib.Path) -> int:
    container = build_container()
    try:
        known: dict[str, str] = {}
        evidence_for: dict[str, tuple[str, ...]] = {}
        for paper_id in await container.knowledge_repository.list_ids():
            stored = await container.knowledge_repository.get(paper_id)
            if stored is None:
                continue
            for obj in stored.value.objects:
                known[obj.id] = obj.name
                evidence_for[obj.id] = tuple(item.id for item in obj.evidence)

        missing: list[str] = []
        queries: list[GoldQuery] = []
        for query_id, text, kinds, judgements, intent in QUERIES:
            unknown = [oid for oid, _, _ in judgements if oid not in known]
            missing.extend(f"{query_id}: {oid}" for oid in unknown)
            usable = [j for j in judgements if j[0] in known]
            evidence_ids = tuple(
                dict.fromkeys(eid for oid, _, _ in usable for eid in evidence_for[oid])
            )
            queries.append(
                GoldQuery(
                    id=query_id,
                    text=text,
                    kinds=kinds,
                    judgements=tuple(
                        GoldJudgement(
                            knowledge_object_id=oid, relevance=relevance, rationale=rationale
                        )
                        for oid, relevance, rationale in usable
                    ),
                    relevant_evidence_ids=evidence_ids,
                    notes=f"intent={intent}; derived by reading the corpus; awaiting human review",
                )
            )

        if missing:
            # Loud, not silent: a stale id would otherwise depress every arm equally and
            # look like a retrieval result.
            print("object ids not present in the corpus:", file=sys.stderr)
            for item in missing:
                print(f"  {item}", file=sys.stderr)
            return 1

        gold = GoldSet(
            version="v2",
            corpus_description=(
                f"{len(await container.knowledge_repository.list_ids())} validated papers "
                f"from storage/papers/raw/manual, {len(known)} knowledge objects"
            ),
            queries=tuple(queries),
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        gold.save(out)

        by_intent: dict[str, int] = {}
        for _, _, _, _, intent in QUERIES:
            by_intent[intent] = by_intent.get(intent, 0) + 1
        print(f"wrote {out}: {len(gold.queries)} queries, {len(gold.drafts)} draft")
        print(f"judgements: {sum(len(q.judgements) for q in gold.queries)}")
        for intent, count in sorted(by_intent.items()):
            print(f"  {intent:<16}{count}")
        return 0
    finally:
        await container.aclose()


if __name__ == "__main__":
    argv = sys.argv[1:]
    destination = pathlib.Path(argv[argv.index("--out") + 1]) if "--out" in argv else GOLD_PATH
    raise SystemExit(asyncio.run(main(destination)))
