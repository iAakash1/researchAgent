# Research Paper Library — GA3 Final Year Project

**Project:** *Interoperability Under Stress: Measuring and Controlling Retry Amplification in Heterogeneous AI-Agent Systems*
**Purpose:** preparation for the expert meeting — defending novelty, motivation, research questions and methodology.
**Built:** 8 August 2026 · 18 files, all obtained from official or author-authorised sources.

---

## ⚠️ Citation errors found while building this library — fix before the meeting

Downloading the originals surfaced three problems in the synopsis reference list. **Two are real errors.**

| Ref | Problem | Correct form |
|---|---|---|
| **[2]** | **WRONG AUTHOR LIST.** Synopsis says *"L. Huang, M. Garrett, and T. Zhu."* There is no author named M. Garrett on this paper. | L. Huang, M. Magnusson, A. Bangalore Muralikrishna, S. Estyak, R. Isaacs, A. Aghayev, T. Zhu, and A. Charapko |
| **[3]** | **MISSING CO-AUTHOR.** The Lawrence Berkeley Laboratory version lists **Michael J. Karels** (UC Berkeley) alongside Van Jacobson. The SIGCOMM '88 proceedings version is conventionally cited as Jacobson alone. | Decide per your citation style; if you cite the LBL PDF in this folder, include Karels. |
| **[11]** | **AUTHORS NOW AVAILABLE** — previously cited by title only. | Y. Xie, C. Zhu, X. Zhang, T. Zhu, D. Ye, M. Qi, H. Chen, and W. Zhou (City University of Macau; Minzu University of China) |

**Verified correct:** [1] Bronson, Aghayev, Charapko, Zhu — HotOS '21 ✓ · [6] Wang, Yu, Lyu ✓ · [7] Du et al., **ICML 2026, PMLR 306, Seoul** — venue claim confirmed on the paper's first page ✓ · [10] Cemri et al. ✓ · [12] Fernandez ✓ · [13] Raskar et al. ✓

**One thing to know about [13]:** the NANDA paper is watermarked **"V0.3, Work in Progress, Request for Comments. Draft."** It is not peer-reviewed. If an expert asks you to justify a claim about NANDA, say so before they do.

---

## Reading plan

| Priority | Files | Total pages | Suggested time |
|---|---|---|---|
| ⭐⭐⭐⭐⭐ **P1 — read completely** | 01, 02, 04, 05a, 05b, 06, 07, 10, 13 | ~335 | ~14 h |
| ⭐⭐⭐⭐ **P2 — read carefully** | 03, 14, 15, 16, 17 | ~46 | ~3 h |
| ⭐⭐⭐ **P3 — read key sections** | 08, 09, 11, 12 | ~124 | ~2.5 h |

**If you only have one day:** 01 → 02 (§2 only) → 05a → 09 → 06 → 07. That is the shortest path to defending the motivation and the novelty boundary.

### Two deviations from the priority list you supplied

- **[10] MAST raised from P2 → P1.** It is not supporting material; it is a *methodology* dependency. Your fault injector selects scenarios using it, and your synopsis explicitly states MAST modes are semantic rather than injectable. An expert who knows MAST will challenge that choice directly. You must be able to defend it.
- **[14] Google ARD lowered from P1 → P2.** It is a 7-page announcement supporting one positioning sentence in Related Work, with no methodological bearing. It is short enough to read fully anyway.

---

# P1 — Must Read Completely

---

## 01 · Metastable Failures in Distributed Systems

**Citation** — N. Bronson, A. Aghayev, A. Charapko, and T. Zhu, "Metastable Failures in Distributed Systems," *Proc. Workshop on Hot Topics in Operating Systems (HotOS '21)*, 2021.
**Source** — sigops.org (official HotOS proceedings mirror) · **7 pages** · **~1 h**

**Why included.** This is the single paper your project's framing rests on. Synopsis reference [1], cited in Related Work and in mathematical model (iv).

**Most important sections.** §2 (the framework: trigger, amplification, sustaining effect) — this is the whole paper for your purposes. §3 examples. §4 on why prediction is hard.

**Key takeaways to extract.**
1. The three-part anatomy: a *stable* state, a *trigger*, and a *sustaining effect* that keeps the system degraded after the trigger is gone.
2. Metastable failure is defined by *persistence after trigger removal* — this is exactly the operational criterion in your RQ2.
3. Retry-driven workload amplification is named as a canonical sustaining effect.
4. These failures are "black swan" events — invisible in advance, obvious in hindsight.
5. Capacity degradation and load amplification are the two families of sustaining effect.
6. The authors state that building systems robust to *unknown* metastable failures "remains an open problem."
7. The framework is descriptive, not predictive — it does not tell you where the boundary is.

**Relation to our project.** Supplies the vocabulary (trigger / sustaining effect / metastable state) and the definition your RQ2 operationalises for agent stacks.

**Effect on our novelty — STRENGTHENS.** It establishes the phenomenon as real and important while explicitly leaving systematic prediction open. Their assumption that retry policy lives in deterministic code is precisely the assumption your model-reasoning layer violates.

**You must be able to answer.**
- What are the three components of a metastable failure, and what is the sustaining effect in *our* system?
- Why is metastability different from ordinary overload?
- Bronson et al. say prediction is an open problem — are you claiming to solve it? *(No. You are locating a boundary empirically in one setting.)*

---

## 02 · Metastable Failures in the Wild

**Citation** — L. Huang, M. Magnusson, A. Bangalore Muralikrishna, S. Estyak, R. Isaacs, A. Aghayev, T. Zhu, and A. Charapko, "Metastable Failures in the Wild," *Proc. 16th USENIX Symposium on Operating Systems Design and Implementation (OSDI '22)*, Carlsbad, CA, 2022.
**Source** — USENIX open access · **19 pages** · **~2 h** · ⚠️ *synopsis author list is wrong — see table above*

**Why included.** Synopsis reference [2]. Provides the empirical grounding and the study methodology your Phase 3 imitates.

**Most important sections.** §2 (extended framework, refined from paper 01). The public-incident study. The experimental sections showing how they reproduced metastability in a controlled testbed — **this is the closest published template for your own experimental design.**

**Key takeaways to extract.**
1. How the Bronson framework was refined after contact with real incidents.
2. Orange/red-line terminology for the boundary between vulnerable and stable operation.
3. How they *reproduced* metastability in a lab — read this for method, not results.
4. Which triggers appear most often in practice.
5. The role of retry policy and timeout configuration in sustaining effects.
6. Evidence that these failures are common rather than exotic.

**Relation to our project.** Your Phase 3 does for agent stacks what this paper did for conventional distributed systems.

**Effect on our novelty — STRENGTHENS, but read defensively.** It is the strongest possible precedent for your method. It is also the paper an expert will use to ask *"what is left for you to do?"* Your answer must be the model-reasoning retry layer and token-denominated cost, neither of which exists here.

**You must be able to answer.**
- How did they induce metastability experimentally, and how does your fault injector differ?
- What is your equivalent of their orange/red line?
- What in your setting is *not* covered by this paper?

---

## 04 · Agent2Agent (A2A) Protocol Specification v1.0

**Citation** — Linux Foundation, "Agent2Agent (A2A) Protocol Specification, Version 1.0.0." https://a2a-protocol.org/latest/specification/
**Source** — official specification site, captured 8 Aug 2026 · **143 pages** · **~4 h** (skim-read most, close-read the lifecycle)

**Why included.** Synopsis reference [4]. One of the two protocols under study, and the source of the terminal-state immutability claim in RQ3.

**Most important sections.** §1 Introduction and key goals · **the Task lifecycle and TaskState enumeration — read this closely, it is load-bearing for RQ3** · §3.4 multi-turn interaction · transports and the three bindings · streaming and `SubscribeToTask` semantics · push notifications.

**Key takeaways to extract.**
1. The eight task states and which four are terminal.
2. The exact normative sentence forbidding messages to terminal tasks.
3. `contextId` vs `taskId` semantics.
4. That three bindings exist (JSON-RPC, gRPC, HTTP+JSON) and must be behaviourally equivalent — relevant to your heterogeneity claim.
5. What `SubscribeToTask` returns on reconnect, and what it does *not* replay.
6. Which retry or recovery behaviour the spec defines — **it defines none**; be able to say that from memory.

**Relation to our project.** RQ3 tests the cost of terminal-state immutability. You cannot defend RQ3 without being able to quote this spec.

**Effect on our novelty — STRENGTHENS.** The absence of any retry, checkpoint or resumption primitive is the gap your project measures.

**You must be able to answer.**
- Quote the terminal-state rule. What exactly happens to work in progress when a task fails?
- Why can a client not simply resume?
- Does A2A define any congestion or backpressure signal? *(No.)*

---

## 05a · MCP Specification 2026-07-28 — Key Changes  ·  05b · Specification Overview

**Citation** — Agentic AI Foundation, "Model Context Protocol Specification, Revision 2026-07-28." https://modelcontextprotocol.io/specification/2026-07-28/
**Source** — official specification site · **6 + 5 pages** · **~1.5 h**

**Why included.** Synopsis reference [5]. The second protocol under study and the source of the "resumability removed" claim in RQ3.

**Most important sections.** 05a §1 Major changes — items 1, 2 and 5 especially. Read 05a *first*; 05b is orientation.

**Key takeaways to extract.**
1. Protocol-level sessions and `Mcp-Session-Id` removed (SEP-2567); cross-call state now uses server-minted handles passed as ordinary tool arguments.
2. The `initialize` handshake removed; MCP is now stateless, with version and capabilities in `_meta`.
3. `server/discover` added as a mandatory RPC.
4. **SSE stream resumability and message redelivery removed — a broken stream loses the in-flight request and clients MUST re-issue as a new request.** This is the sentence RQ3 depends on. Memorise it.
5. Tasks moved out of core into the `io.modelcontextprotocol/tasks` extension.
6. Multi Round-Trip Requests replace server-initiated requests.
7. OpenTelemetry trace-context conventions in `_meta` — this is what makes your instrumentation possible.

**Relation to our project.** Directly motivates RQ3, and item 7 underwrites your instrumentation design.

**Effect on our novelty — STRENGTHENS.** A standards body deliberately raised per-retry cost in 2026 and has not addressed retry dynamics.

**You must be able to answer.**
- What exactly does a client do when an MCP stream breaks mid-request?
- Why does statelessness make amplification worse rather than better?
- Where does MCP now carry trace context, and how will you use it?

---

## 06 · AI-NativeBench

**Citation** — Z. Wang, G. Yu, and M. R. Lyu, "AI-NativeBench: An Open-Source White-Box Agentic Benchmark Suite for AI-Native Systems," arXiv:2601.09393, 2026. (Sun Yat-sen University; CUHK)
**Source** — arXiv · **39 pages** · **~2.5 h**

**Why included.** Synopsis reference [6]. Supplies the two numbers your motivation rests on.

**Most important sections.** The methodology (white-box, agentic spans as first-class citizens in distributed traces) · the four architectural variants (CrewAI / +MCP / +A2A / +H-A2A) · the results on retry rate and token economics · **the finding that LLM computation is 86.9–99.9% of execution time.**

**Key takeaways to extract.**
1. Failed workflows consume *more* resources than successful ones — systems exhaust retry budgets instead of failing fast. **This is your motivating observation.**
2. LLM computation dominates end-to-end time (86.9–99.9%), so protocol overhead is secondary — which is *why* amplification matters through repeated model invocation, not protocol chatter.
3. Heterogeneous A2A composition imposes a measurable reliability cost.
4. MCP refactoring reduced retry rates without accuracy loss.
5. Treating agentic spans as trace citizens — methodologically close to your instrumentation.
6. 21 system variants, 7 LLMs — the scale you are *not* attempting.

**Relation to our project.** The strongest existing evidence that the phenomenon is real, and the source of your "amplification is economic" argument.

**Effect on our novelty — MIXED. Read this most carefully of all.** It strengthens motivation but it is also the nearest prior measurement. Your defence: they *observed* wasted retries as one finding among many; they did not characterise amplification, decompose it by layer, test for metastability, or propose control.

**You must be able to answer.**
- What did they measure that you are not re-measuring?
- If LLM compute is 86.9–99.9% of runtime, why does protocol-layer work matter at all?
- Why is their "failed workflows cost more" finding not already your result?

---

## 07 · ProtocolBench

**Citation** — H. Du, J. Su, J. Li, L. Ding, Y. Yang, P. Han, X. Tang, K. Zhu, and J. You, "ProtocolBench: Which LLM MultiAgent Protocol to Choose?", *Proc. 43rd International Conference on Machine Learning (ICML)*, Seoul, PMLR 306, 2026.
**Source** — arXiv · **47 pages** · **~2.5 h** · ✅ ICML 2026 venue verified on page 1

**Why included.** Synopsis reference [7]. The closest published benchmark, and a peer-reviewed one.

**Most important sections.** The four evaluation axes · **the Fail-Storm Recovery scenario — this is the nearest published work to your RQ2 and you must know it in detail** · ProtocolRouter · the Streaming Queue results.

**Key takeaways to extract.**
1. Four axes: task success, end-to-end latency, message/byte overhead, robustness under failures.
2. Completion time varies up to 36.5% across protocols; mean end-to-end latency differs by 3.48 s.
3. Fail-Storm Recovery — how they construct it, and what they measure.
4. ProtocolRouter reduces Fail-Storm recovery time by up to 18.1%.
5. Protocols covered: A2A, ACP, ANP, Agora.
6. Protocol choice is treated as the independent variable — *not* the layering of retries within one stack.

**Relation to our project.** This is your primary baseline and your primary novelty threat.

**Effect on our novelty — WEAKENS unless you draw the line precisely.** They treat failure storms as an evaluation *scenario* to compare protocols; you treat amplification as a *phenomenon* to characterise within a stack. Their independent variable is protocol choice; yours is offered load and retry-layer configuration. Rehearse this distinction until it is one sentence.

**You must be able to answer.**
- How does a Fail-Storm differ from what you are inducing?
- Why is your work not a subset of ProtocolBench?
- Could you have used their harness instead of building one? *(Be honest about why not.)*

---

## 10 · Why Do Multi-Agent LLM Systems Fail? (MAST)

**Citation** — M. Cemri, M. Z. Pan, S. Yang, L. A. Agrawal, B. Chopra, R. Tiwari, K. Keutzer, A. Parameswaran, D. Klein, K. Ramchandran, M. Zaharia, J. E. Gonzalez, and I. Stoica, "Why Do Multi-Agent LLM Systems Fail?", arXiv:2503.13657, 2025. (UC Berkeley; Intesa Sanpaolo)
**Source** — arXiv · **47 pages** · **~2 h** · *raised to P1 — see justification above*

**Why included.** Synopsis reference [10]. Your fault-injection scenarios are selected using this taxonomy.

**Most important sections.** The MAST taxonomy itself (3 categories, 14 modes) · the construction and validation methodology (150 traces, expert annotators, inter-annotator agreement) · MAST-Data (1600+ traces, 7 frameworks).

**Key takeaways to extract.**
1. Three categories and all fourteen failure modes — know the category names at minimum.
2. The taxonomy was validated with expert annotators and reported agreement — the standard your own classification work will be held to.
3. Which modes are *semantic* (task derailment, premature termination) and therefore **not injectable at a transport boundary**.
4. MAS performance gains on benchmarks are often minimal — useful context.
5. Seven frameworks studied — the breadth you are not attempting.

**Relation to our project.** Your synopsis states MAST modes are used to *select realistic scenarios*, not as injection primitives. This is a deliberate, defensible position — **but only if you can articulate it.**

**Effect on our novelty — NEUTRAL. It is a defensibility risk, not a novelty risk.** An expert who knows MAST may say "you cannot inject these." Your answer: correct, and we do not — we inject transport and tool faults, and use MAST to decide which resulting agent-level scenarios are realistic.

**You must be able to answer.**
- Name the three categories.
- Why can you not simply "inject" MAST failure modes?
- What exactly does your fault injector inject, and how does MAST inform it?

---

## 13 · Beyond DNS: NANDA Index and Verified AgentFacts

**Citation** — R. Raskar, P. Chari, J. Zinky, S. Wang, R. Singhal, R. Lincourt, M. Lambe, J. J. Grogan, R. Ranjan, S. Gupta, R. Bala, A. Joshi, A. Singh, A. Chopra, D. Stripelis, B. B, S. Kumar, and M. Gorskikh, "Beyond DNS: Unlocking the Internet of AI Agents via the NANDA Index and Verified AgentFacts," arXiv:2507.14263, 2025. (MIT Media Lab / Project NANDA)
**Source** — arXiv · **22 pages** · **~1.5 h** · ⚠️ **marked V0.3 Work in Progress / RFC Draft — not peer-reviewed**

**Why included.** Synopsis reference [13]. Supports the Rationale's central claim that the Internet of Agents has discovery, identity and trust layers but no flow-control layer. Your HOD specifically asked you to study NANDA.

**Most important sections.** The abstract's framing (billions to trillions of agents straining DNS-centred identity and discovery) · the index architecture · AgentFacts and multi-endpoint routing · **what is explicitly out of scope.**

**Key takeaways to extract.**
1. NANDA layers *above* MCP/A2A rather than replacing them.
2. A lean index resolves to cryptographically verifiable AgentFacts.
3. AgentFacts support multi-endpoint routing and load balancing — **note this is address-level, not execution-state-level.**
4. Sub-second revocation and key rotation; schema-validated capability assertions.
5. NANDA defines no task lifecycle, no execution semantics, and no load or stability mechanism.
6. It is a draft RFC, and the appropriate confidence to place in it is correspondingly lower.

**Relation to our project.** Establishes that the layers around yours are being actively built while yours is not — the core of your positioning.

**Effect on our novelty — STRENGTHENS.** NANDA is genuinely orthogonal: it answers *which agent and at what address*, not *what happens to work in flight under load*.

**You must be able to answer.**
- What layer does NANDA occupy, and why is your work not competing with it?
- AgentFacts supports failover — how is that different from what you do? *(It re-resolves a name to a different endpoint. It does not preserve or coordinate execution.)*
- Would your contribution still matter if NANDA succeeded completely? *(Yes — arguably more.)*

---

# P2 — Read Carefully

---

## 03 · Congestion Avoidance and Control

**Citation** — V. Jacobson and M. J. Karels, "Congestion Avoidance and Control," Lawrence Berkeley Laboratory, November 1988. Published version: *Proc. ACM SIGCOMM*, pp. 314–329, 1988. ⚠️ *see co-author note above*
**Source** — ee.lbl.gov (author institution) · **25 pages** · **~1.5 h**

**Why included.** Synopsis reference [3]. Supplies the historical argument that layered networks need an explicit stability mechanism, not merely correct delivery.

**Most important sections.** The introduction and the packet-conservation principle. §1–3. You do not need the appendices.

**Key takeaways to extract.**
1. Congestion collapse struck an already-working network — correctness was never the problem.
2. The packet-conservation principle as the basis of stability.
3. The failure was in *implementations of* the transport protocol, not the protocol design.
4. Slow-start, RTT variance estimation, exponential backoff.
5. Stability had to be added years after the network functioned — your analogy.

**Relation to our project.** The rhetorical backbone of your motivation: agent networks today are pre-1986.

**Effect on our novelty — STRENGTHENS the framing, but handle with care.** An unsympathetic expert may say "so this is just congestion control again." Your answer is the non-deterministic retry layer and token-denominated cost — neither exists in Jacobson's setting.

**You must be able to answer.**
- Why did congestion collapse happen a decade after the network worked?
- What is the agent-network analogue of packet conservation — and is there one?
- What does Jacobson's setting *lack* that yours has?

---

## 14 · Agentic Resource Discovery (ARD) Specification

**Citation** — J. Bu and S. Krishnan, "Announcing the Agentic Resource Discovery specification," Google Developers Blog, 17 June 2026.
**Source** — Google Developers Blog · **7 pages** · **~30 min**

**Why included.** Synopsis reference [14], supporting the Related Work claim that discovery is actively being built.

**Key takeaways.** ARD answers three questions — where a capability lives, which to use, and how to verify it is safe. `ai-catalog.json` published per domain, plus crawling registries. Open specification, Apache 2.0. Domain ownership functions as the identity anchor. It defines nothing about execution, load or stability.

**Relation to our project.** One positioning sentence. Low methodological weight.
**Effect on novelty — STRENGTHENS mildly** (another discovery-layer effort that leaves flow control untouched).

**You must be able to answer.** What problem does ARD solve, and why is it not your problem?

---

## 15 · gRPC Retry Design

**Citation** — gRPC Authors, "gRPC Retry Design: Retry Throttling." https://grpc.io/docs/guides/retry/
**Source** — official gRPC documentation · **3 pages** · **~30 min**

**Why included.** Synopsis reference [15]. Your baseline for retry budgets and client-side throttling.

**Key takeaways.** How gRPC saves call history for replay. Retry policy configuration. **Retry throttling — the token-bucket mechanism that is the direct ancestor of your workflow-scoped budget.** Transparent retry vs configured retry. Hedging. The stated best practice: know which operations are safe to retry, bound attempts, monitor retry metrics.

**Relation to our project.** This is the mechanism you are adapting; your delta is *workflow scope* and *token denomination*.

**Effect on novelty — WEAKENS if you overclaim.** Retry budgets are prior art and well understood. Never present the budget itself as novel.

**You must be able to answer.** How does gRPC throttling work, and what precisely does your mechanism add? Why does per-channel throttling not solve the agent case?

---

## 16 · OpenTelemetry — Context Propagation

**Citation** — Cloud Native Computing Foundation, "Context propagation," OpenTelemetry Documentation. https://opentelemetry.io/docs/concepts/context-propagation/
**Source** — official OTel documentation · **6 pages** · **~30 min**

**Why included.** Synopsis reference [16]. The mechanism that makes cross-layer attribution possible.

**Key takeaways.** Context carries trace and span IDs across process and network boundaries. Propagators serialise and deserialise context. Parent-child span relationships build causal structure. W3C TraceContext is the standard format — and MCP's 2026-07-28 revision adopts it in `_meta`.

**Relation to our project.** Your instrumentation at layers L1–L3 depends entirely on this. Note the honest limit: **it cannot instrument the model-reasoning layer**, which is why you use residual attribution.

**Effect on novelty — NEUTRAL** (enabling technology).

**You must be able to answer.** How does a span at the HTTP layer get associated with the task that caused it? Why can OTel not directly measure model-reasoning retries?

---

## 17 · LangGraph — Persistence and Durable Execution

**Citation** — LangChain, "Persistence," LangGraph Documentation. https://langchain-ai.github.io/langgraph/concepts/persistence/
**Source** — official LangGraph documentation · **5 pages** · **~30 min**

**Why included.** Synopsis reference [17]. One of your two frameworks, and the source of the per-node retry configuration claim.

**Key takeaways.** Checkpointers persist thread-scoped graph state at each step. Stores persist cross-thread data. Checkpoints support resumption after interruption, time travel and human-in-the-loop. **Checkpointing is a save point — there is no automatic failure detection, no watchdog, no supervisor.** Retry configuration is per-node.

**Relation to our project.** Establishes that retry policy at L3 is configured per-node, independently of L1/L2 — the structural fact your project exploits.

**Effect on novelty — STRENGTHENS.** LangGraph provides no cross-layer coordination and no workflow-scoped budget.

**You must be able to answer.** Where does LangGraph's retry configuration live, and what is it unaware of? Why is a checkpointer not a solution to amplification?

---

# P3 — Read Key Sections Only

---

## 08 · The 2026 MCP Roadmap

**Citation** — Agentic AI Foundation, "The 2026 MCP Roadmap," 9 March 2026, D. Soria Parra (Lead Maintainer). https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/
**Source** — official MCP blog · **6 pages** · **~20 min**

**Read for one thing:** the statement that early production use surfaced lifecycle gaps — **retry semantics when a task fails transiently, and expiry policies for result retention.** That single sentence is your evidence that the standards body knows retry semantics are undefined.

**Also worth noting.** The four 2026 priority areas (transport scalability, agent communication, governance, enterprise readiness) and the SEP process.

**Effect on novelty — STRENGTHENS.** Confirms the gap is acknowledged upstream and still open. **Read alongside 09.**

**You must be able to answer.** What has the MCP project said about retries, and what have they *not* said? *(They address retry semantics — correctness. Not retry dynamics — stability.)*

---

## 09 · A2A Issue #1987 — Idempotency & Safe Retries

**Citation** — A2A Project, "[Epic] Idempotency & safe retries," Issue #1987, opened 25 June 2026 by Tehsmash. Labels: `enhancement`, `v1.1-candidate`. https://github.com/a2aproject/A2A/issues/1987
**Source** — GitHub · **3 pages** · **~15 min** · *read in full, it is short and important*

**Key takeaways.** The problem: a client that crashes after sending but before persisting `taskId`/`contextId` cannot learn whether a task started, and has no safe retry. There is no transport-level deduplication. Acceptance criteria require a defined idempotency mechanism, server-side deduplication semantics, consistency across all three bindings, and security constraints on idempotency keys. Consolidates #928 (the problem) and #1893 (an `Idempotency-Key` header proposal).

**Relation to our project.** **This is your sharpest single piece of evidence for the correctness-versus-dynamics distinction.** The A2A project is actively working on making retries *safe*. Nobody is working on making them *bounded*.

**Effect on novelty — STRENGTHENS substantially.** Be ready to state it crisply: idempotency prevents duplicate *effects*; it does nothing about duplicate *load*.

**You must be able to answer.**
- If A2A ships idempotency keys in v1.1, is your project obsolete? *(No — and this is the most likely question you will be asked. Rehearse it.)*
- What is the difference between a safe retry and a bounded retry?

---

## 11 · From Spark to Fire: Error Cascades in LLM-Based Multi-Agent Collaboration

**Citation** — Y. Xie, C. Zhu, X. Zhang, T. Zhu, D. Ye, M. Qi, H. Chen, and W. Zhou, "From Spark to Fire: Modeling and Mitigating Error Cascades in LLM-Based Multi-Agent Collaboration," arXiv:2603.04474, 2026. (City University of Macau; Minzu University of China)
**Source** — arXiv · **20 pages** · **~45 min** · *author list newly recovered — update your synopsis*

**Read for the distinction, not the content.** They model collaboration as a directed dependency graph and derive an early-stage risk criterion for **amplification of semantic error** — minor inaccuracies solidifying into system-level false consensus. Note the word *amplification* appears in both their work and yours with **entirely different referents**.

**Effect on novelty — NEUTRAL, but a trap.** An expert skimming titles may think this is your paper. Be ready in one sentence: *they propagate wrong answers; we propagate load.*

**You must be able to answer.** What is amplified in their work versus yours? Could their dependency-graph model apply to load? *(Interesting question — think about it before the meeting.)*

---

## 12 · Agent Control Protocol (ACP)

**Citation** — M. Fernandez, "Agent Control Protocol ACP v1.30: Admission Control for Agent Actions," arXiv:2603.18829, April 2026. TraslaIA. DOI: 10.5281/zenodo.19672575.
**Source** — arXiv · **95 pages** · **~45 min** (abstract, architecture, and the admission-control mechanism only — do not read all 95 pages)

**Read for the distinction.** ACP is a temporal admission-control protocol enforcing behavioural properties over execution traces, combining static risk scoring with stateful signals. It blocks execution based on deterministic, history-aware risk scoring. **It gates actions for safety and policy compliance — not for stability under overload.**

**Key points.** Admission control as a concept applied to agents. The LedgerQuerier abstraction separating decision logic from state. Hard enforcement rather than advisory alerting.

**Effect on novelty — NEUTRAL if framed correctly, WEAKENS if ignored.** It is the closest thing to a control layer for agents in the literature. Your distinction is the objective function: ACP asks *should this action be permitted?*; you ask *can the system absorb this load?*

**You must be able to answer.** Both projects insert a control point above the agent — what makes them different? Could ACP's mechanism be repurposed for load shedding?

---

## Expected challenges — rehearse these

The five hardest questions, in the order you are most likely to face them:

1. **"Isn't this just microservice retry amplification?"** → The fourth retry layer is inside model reasoning: undeclared, unconfigurable, non-deterministic. And retry cost is token-denominated, so amplification is economic, not merely latency-related.
2. **"ProtocolBench already tested failure storms."** → Their independent variable is protocol choice; ours is offered load and retry-layer configuration. They compare protocols under a scenario; we characterise a phenomenon within a stack.
3. **"AI-NativeBench already found retry waste."** → As one observation among many. They did not decompose it by layer, test for metastability, or propose control.
4. **"A2A will ship idempotency keys in v1.1 — you're obsolete."** → Idempotency prevents duplicate *effects*, not duplicate *load*. Both epics could ship and our measurement would be unchanged.
5. **"Your mock backend is deterministic, but you claim non-determinism is the novelty."** → We calibrate the reasoning layer's retry propensity on commercial endpoints and replay it as a measured stochastic process. The non-determinism is represented by observed behaviour, not removed.

---

## Provenance

Every file was retrieved from an official or author-authorised source: arXiv (6), USENIX open access (1), ACM SIGOPS HotOS proceedings mirror (1), Lawrence Berkeley Laboratory (1), official specification and documentation sites (8), GitHub (1). No paywalled material was bypassed and no unofficial mirrors were used. Web-based specifications were captured as PDF on 8 August 2026 and reflect their state on that date — the A2A and MCP specifications are living documents and may change before your meeting.
