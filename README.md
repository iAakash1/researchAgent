# ResearchAgent

Local-first multi-agent research intelligence platform. Give it a research goal; it plans,
searches literature, retrieves and parses papers, verifies every claim against the source
text, finds gaps, critiques its own report, and exports a literature review with evidence.

Everything runs on your machine through [Ollama](https://ollama.com). No API keys, no data
leaving the host.

> **Status: v0.3** — a research goal becomes a plan, then real papers: six providers
> (arXiv, OpenAlex, Crossref, Semantic Scholar, PubMed and your own PDF collection)
> searched, deduplicated, ranked and stored. PDF parsing lands in v0.4.
> See [ROADMAP](#roadmap).

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
| v0.4 | PDF parsing and section-aware chunking |
| v0.5 | Hybrid RAG over Qdrant with reranking and citation preservation |
| v0.6 | Verification agent — every extracted claim checked against source text |
| v0.7 | Knowledge graph (Neo4j) |
| v0.8 | Synthesis, gap discovery, reviewer self-critique loop, persistent memory |
| v0.9 | Web UI |
| v1.0 | Evaluation framework and reproducible experiments |

## License

Apache-2.0
