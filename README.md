# ResearchAgent

A local-first research system that turns a folder of papers into evidence you can audit —
and refuses to state a conclusion it cannot trace back to a page.

## What it does

Point it at research PDFs. It parses them, extracts typed knowledge (methods, datasets,
metrics, results, limitations, future work), grounds every extracted claim in the sentence
that supports it, and builds lexical, semantic and graph indexes over the result.

Ask a research question and five agents go to work under a LangGraph loop: one finds
evidence, one reasons over it, one attacks the conclusions, one decides what survives.
What comes out is a set of findings, each carrying citations that resolve to a page and
paragraph in an original PDF — or an explicit statement that the corpus cannot answer the
question.

## Why it exists

Ask an LLM a research question and you get fluent prose with plausible citations. Some are
real. Checking which is the actual work, and the system that produced them gives you no
help with it.

ResearchAgent inverts that. The language model is treated as an untrusted component: it
proposes, and everything it proposes has to survive a chain of validators that can reject
it. A claim with no locatable evidence does not become a quiet mistake in paragraph four —
it never becomes a finding at all.

## Architecture

```
  RESEARCH PAPERS (PDF)
          │
          ▼
     INGESTION            metadata, dedup, validation
          │
          ▼
   DOCUMENT PARSING       pages → sections → paragraphs
          │
          ▼
  KNOWLEDGE EXTRACTION    typed facts, each grounded in a quote
          │
          ▼
     EVIDENCE STORE       quote + page + section + paragraph
          │
   ┌──────┴───────┬───────────────┐
   ▼              ▼               ▼
 BM25         EMBEDDINGS    KNOWLEDGE GRAPH
   └──────┬───────┴───────────────┘
          ▼
       RETRIEVAL           deterministic · bm25 · semantic · hybrid
          │
          ▼
   ┌─────────────────────────────────────────┐
   │        LANGGRAPH AGENT LOOP             │
   │                                         │
   │  Retrieval → Reasoning → Verification   │
   │      ▲                        │         │
   │      └──── insufficient ──────┤         │
   │           contradicted ───────┘         │
   │                       │                 │
   │                    Reviewer             │
   └───────────────────────┬─────────────────┘
                           ▼
                   AUDITED FINDING
```

The one-line version: **agents propose, evidence supports, validators decide.**

## The agents

Five, each with one job, all orchestrated by LangGraph — no agent calls another.

| Agent | Decides | Output |
|---|---|---|
| **Planner** | what needs investigating | research questions |
| **Retrieval** | how to search, and whether the result is enough | evidence bundles |
| **Reasoning** | what the evidence supports | findings with citations |
| **Verification** | whether the claim survives scrutiny | verdict + its own citations |
| **Reviewer** | what is accepted | accept / revise / reject |

The roster is a plain dict in [`agents/registry.py`](researchagent/agents/registry.py).

**LangGraph owns the control flow.** After verification it routes on the verdict:
`INSUFFICIENT_EVIDENCE` goes back to retrieval (find more), `CONTRADICTED` goes back to
reasoning (the claim is wrong, not under-evidenced), `VERIFIED` goes to review. Iterations,
tool calls and tokens are all budgeted, and every exit records a termination reason.

**Verification is adversarial by construction**, not by instruction:

- provenance is resolved *before* the model is asked anything — citations that lead
  nowhere return `UNVERIFIABLE` without an opinion being sought
- the verifier sees evidence the finding did *not* cite, because contradicting material is
  by definition what was left out
- a `VERIFIED` verdict with no citations of its own is downgraded, not accepted

## Zero-trust evidence

The guarantee is structural, enforced by the models rather than by review:

- a `KnowledgeObject` **cannot be constructed** without evidence
- a `ResearchFinding` **cannot be constructed** without a `Citation`
- a `Citation` **cannot be constructed** without an `EvidenceBundle` id

So the chain from claim to page is total. When the reasoning agent emits a citation id that
does not resolve, the id is dropped; when a claim loses all its citations it becomes a
*hypothesis* and is counted, never silently attached to some other evidence.

That count is the system's own fabrication rate, and it is reported rather than hidden.

## Data pipeline

This is a data engineering project as much as an LLM one. The pipeline is batch,
deterministic, configuration-driven and reproducible:

| Stage | What happens |
|---|---|
| **Ingest** | PDFs discovered, metadata extracted, duplicates merged on DOI/title |
| **Parse** | PyMuPDF → positioned text blocks → sections, references, figures |
| **Validate** | non-research documents rejected with reasons, not silently processed |
| **Extract** | six extractors produce typed knowledge, one kind each |
| **Ground** | every model quote located in the source text; what cannot be located is discarded |
| **Index** | BM25, embeddings (Ollama `nomic-embed-text`), knowledge graph |
| **Retrieve** | four strategies over the same corpus, benchmarked against a gold set |
| **Serve** | agents consume `EvidenceBundle`s, never raw text |

Failure is data, not an exception: one unreadable PDF costs one paper and the run stays
inspectable. Extraction counters are persisted with the knowledge so a cached paper
reports the same grounding rate it did when it was extracted.

**Measured on the committed corpus:** 18 PDFs → 12 valid research papers, 6 correctly
rejected (specs, roadmaps, a GitHub issue). 296 extraction proposals → 239 grounded
(**0.807 grounding rate**) → 222 validated knowledge objects → 137 evidence records.

## Knowledge graph

233 nodes, 271 edges over the 12-paper corpus, **100% of edges carrying provenance** — an
edge with no locatable evidence is rejected with a reason rather than written.

Entities merge across papers so cross-paper questions are answerable (`MIMIC-III` is one
node, not one per paper), but *claim-like* kinds — results, limitations, future work — stay
scoped to their paper. Merging those would collapse a disagreement between two papers into
a self-loop and delete the finding.

The graph is a **derived index**. The knowledge and evidence repositories are the source of
truth; deleting the graph costs a rebuild, never a fact. Three backends behind one port:
in-memory, JSON files (the default — no server needed), and Neo4j.

## Retrieval

Four arms, same corpus, same candidate pool, measured on a 26-query gold set whose
judgements were derived by reading the corpus — **never generated by an LLM**, because a
benchmark labelled by a model measures agreement with that model.

| arm | P@5 | R@10 | MRR | nDCG@5 | evidence coverage |
|---|---:|---:|---:|---:|---:|
| deterministic | 0.354 | 0.466 | 0.666 | 0.412 | 0.321 |
| bm25 | 0.308 | 0.548 | 0.648 | 0.377 | 0.364 |
| **semantic** | **0.600** | **0.738** | **0.910** | **0.698** | **0.462** |
| hybrid | 0.469 | 0.619 | 0.853 | 0.566 | 0.405 |

**Honest reading:** semantic wins clearly and hybrid fusion *dilutes* it — a weight
ablation showed a monotone trend, with pure-dense the best endpoint. The production default
is still `deterministic`, because all 26 gold queries are `draft` status and only reviewed
judgements may back a claim. Nothing was tuned to make hybrid look better.

## A real example

Question actually run against the committed corpus:

> *What do the papers recommend for controlling retry behaviour, and do their
> recommendations agree?*

Accepted finding (Groq GPT-OSS 120B, `evaluation/experiments/cross_b_groq.json`):

> "The papers do not fully agree on the preferred control mechanism for retries; one
> recommends disabling retries entirely while another recommends lowering their priority."

Its two citations resolve independently, in two different papers:

```
Finding F-c4ff18f4
├── Citation → evidence 3b8879b2
│     "You can also disable retries entirely when creating a channel…"
│     manual:15  p.2  §Retry configuration  ¶4
│     → document sha256 703b928142c4 → 15_[P2]_gRPC_Retry_Design.pdf
└── Citation → evidence 3decf172
      "Prioritization: Another way to retain efficiency when a resource is exhausted…"
      manual:01  p.4  §3 Approaches to Handling Metastability  ¶5
      → document sha256 3556cdef28e5 → 01_[P1]_Metastable_Failures_in_Distributed_Systems.pdf
```

## Evaluation

The same question, same corpus, same retrieval config, same budgets — only the reasoning
model differs. Comparability is enforced: `compare_runs.py` refuses to print when two
artifacts disagree on corpus fingerprint, index version or config digest.

| | Ollama `llama3.1:8b` | Groq `openai/gpt-oss-120b` |
|---|---:|---:|
| findings proposed | 19 | 6 |
| **accepted** | **0** | **2** |
| contradicted / insufficient | 3 / 7 | 0 / 0 |
| citation completeness | — | 1.0 |
| unsupported claim rate | — | 0.0 |
| iterations / tool calls | 2 / 33 | 1 / 16 |
| input / output tokens | 54,954 / 2,935 | 33,980 / 7,164 |
| termination | `budget_exhausted` | `all_questions_answered` |

**What this does and does not show.** The 8B local model produced *more* findings and had
*none* survive verification — the pipeline rejected its own output, which is the system
working. The 120B model spent 2.4× more output tokens reasoning but 29% fewer overall,
converging in one iteration. This is one question on one 12-paper corpus: it is a
demonstration that the harness is reproducible, not a benchmark of either model.

**Known limits, stated plainly:** the gold set is unreviewed `draft`; only one of the two
accepted findings is genuinely cross-paper; the retrieval agent has never once chosen graph
expansion over direct retrieval, so that path is tested but unexercised in practice.

## Tech stack

Python 3.12 · LangGraph · Pydantic v2 · FastAPI · PyMuPDF · Ollama · Groq · Qdrant ·
Neo4j · uv · ruff · mypy `--strict` · pytest

Ollama is the default and everything runs offline without an API key. Groq is optional:
both sit behind one `LLMProvider` port, and no agent knows which is in use.

## Running locally

```bash
uv sync --extra dev          # install
make models                  # pull llama3.1:8b + nomic-embed-text via Ollama
cp .env.example .env         # optional: add GROQ_API_KEY to enable the remote provider
make check                   # lint + typecheck + 660 tests
```

Build the corpus and indexes, then ask a question:

```bash
make index                   # metadata sidecars for storage/papers/raw/manual/
make graph                   # materialise the knowledge graph
make benchmark               # compare the four retrieval arms

uv run python scripts/run_experiment.py \
  --alias reasoning --label my_run --build-graph \
  --goal "<research goal>" --question "<research question>"

uv run python scripts/compare_runs.py my_run other_run
```

`--alias reasoning` runs locally on Ollama; `--alias reasoning_remote` uses Groq
GPT-OSS 120B. Everything else is held fixed.

## Project structure

```
researchagent/
  agents/          five agents; registry.py is the whole roster
  workflows/       LangGraph graphs, nodes, guards, routing, budgets
  services/        document · knowledge · evidence · retrieval · graph · tools
  models/          domain objects: Paper, KnowledgeObject, Evidence, Finding, Graph
  schemas/         cross-stage contracts and workflow state
  repositories/    JSON persistence adapters
  integrations/    Ollama, Groq, Qdrant, Neo4j, PyMuPDF, paper sources
  core/            settings, events, validation, errors, ports (interfaces/)
config/            one YAML per subsystem — models, retrieval, graph, budgets, agents
evaluation/        gold set, benchmark results, reproducible experiment artifacts
scripts/           corpus indexing, graph build, benchmark, experiments
storage/           the committed 18-PDF corpus (derived state is gitignored)
```

## Future work

- Review the 26-query gold set so benchmark numbers can back a claim rather than indicate
- Give the retrieval agent a reason to choose graph traversal, or conclude it does not need one
- Persist checkpoints beyond process memory so a run is resumable
- Widen the corpus enough for cross-paper synthesis to be routine rather than singular
