# ResearchAgent

Local-first multi-agent research intelligence platform. Give it a research goal; it plans,
searches literature, retrieves and parses papers, verifies every claim against the source
text, finds gaps, critiques its own report, and exports a literature review with evidence.

Everything runs on your machine through [Ollama](https://ollama.com) by default. No API
keys, no data leaving the host.

An optional external provider is available for the reasoning-heavy stages: set
`GROQ_API_KEY` and point a model alias in `config/models.yaml` at `provider: groq`
(the `reasoning_remote` alias uses GPT-OSS 120B). Both providers sit behind the same
`LLMProvider` port, so nothing above `integrations/` changes. Leave `GROQ_API_KEY` unset
and the system stays fully local and offline — including the test suite.

> **Status: v0.9** — a research goal becomes a plan, then real papers, then validated
> documents, then evidence-backed knowledge: methods, datasets, metrics, results,
> limitations and future work, each traceable to the page and paragraph that states it.
> From v0.9 four agents reason over that evidence in a bounded loop — retrieve, reason,
> verify, review — and a conclusion is only accepted if its citations resolve all the way
> back to a source location. See [ROADMAP](#roadmap).

Run a research question over the local corpus, and compare two reasoning models on it:

```bash
uv run python scripts/run_experiment.py --alias reasoning --label local \
  --goal "<research goal>" --question "<research question>"
uv run python scripts/compare_runs.py local other_run
```

`--alias reasoning` uses Ollama; `--alias reasoning_remote` uses Groq GPT-OSS 120B. The
comparison refuses to run when the two artifacts disagree on corpus, index or
configuration, so only the model can differ.

**Nothing is believed without evidence.** Every quote a model produces is located in the
source PDF before it becomes a fact; what cannot be located is discarded, and the
rejection rate is reported. A `KnowledgeObject` cannot exist without provenance.

**Zero trust.** Every stage assumes the previous one may be wrong. Discovery metadata is
cross-checked against what the PDF says about itself; every extracted fact carries the
page and paragraph it came from; every confidence score names the observation behind it.
Stages exchange validated artefacts, never bare objects.

## Architecture

```
api  →  workflows  →  agents  →  services  →  integrations
                          ↘         ↓             ↑
                        schemas  repositories  core.interfaces
                                     ↓
                                   core
```

Dependencies point inward. Vendor SDKs are only imported inside `integrations/`; everything
else depends on a port in `core/interfaces/`. Swapping Ollama for another runtime, or Qdrant
for another vector store, is a new adapter plus one line of YAML — no agent changes.

## Quick start

```bash
cp .env.example .env
uv sync --extra dev
```

Pull the models named in `config/models.yaml`:

```bash
make models
```

Run the API:

```bash
make dev
```

Then `curl localhost:8000/health/ready` — it reports each provider's health and whether every
configured model is actually pulled. Interactive docs at `localhost:8000/docs`.

Plan a review:

```bash
curl -X POST localhost:8000/research/plan -H 'content-type: application/json' \
  -d '{"goal":"Agentic AI in healthcare","constraints":{"year_from":2022}}'
```

You get research questions, a search strategy, and a ranked set of candidate papers
discovered across every enabled provider. `POST /research/plan/stream` streams the same
run as server-sent events; `GET /research/runs/{id}` reloads it from its checkpoint.

### Your own papers

Drop PDFs into `storage/papers/raw/manual/` and they take part in discovery alongside
arXiv and OpenAlex — matched, ranked and deduplicated against them. Where the same paper
is also indexed online, the records merge, so a local file gains its real DOI, year,
venue and citation count. Files are read only, never moved or modified.

```bash
make index    # write metadata sidecars to storage/papers/metadata/
```

## Full stack with Docker

Brings up Ollama, PostgreSQL, Qdrant, Neo4j and the API:

```bash
docker compose up -d
```

Models must be pulled inside the container once:

```bash
docker compose exec ollama ollama pull qwen3:8b
```

## Configuration

Behaviour lives in YAML, environment lives in `.env`. Changing the model an agent uses is a
one-line edit to `config/models.yaml` or `config/agents.yaml` — never a code change.

```yaml
# config/models.yaml
default: reasoning
models:
  reasoning:
    provider: ollama
    model: qwen3:8b
    params:
      temperature: 0.3
      context_window: 32768
```

Environment variables use the `RESEARCHAGENT_` prefix and `__` for nesting:
`RESEARCHAGENT_OLLAMA__BASE_URL=http://ollama:11434`.

## Development

```bash
make check     # ruff + mypy --strict + pytest
make test
make format
```

## Roadmap

| Version | Delivers |
|---|---|
| v0.1 | Skeleton: config, LLM port, agent contract, API, Docker, CI |
| v0.2 | Planner agent + LangGraph state orchestration, versioned prompts |
| v0.3 | Literature discovery + retrieval: six providers, dedup, ranking, JSON library |
| v0.4 | Document intelligence engine + zero-trust foundation (validation, evidence, guards) |
| v0.5 | Knowledge intelligence: evidence-grounded `KnowledgeObject` extraction |
| v0.6 | Evidence retrieval: section-aware chunking, embeddings, hybrid retrieval |
| v0.7 | Knowledge graph (Neo4j) from extracted facts only |
| v0.8 | Reasoning engine over graph + evidence + documents |
| v0.9 | Verification agent — every claim checked against source evidence |
| v1.0 | Research intelligence platform: reviewer loop, UI, agent evaluation benchmarks |

## License

Apache-2.0
