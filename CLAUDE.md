# CLAUDE.md — ResearchAgent

Read this before implementing anything. Prompts should say: *"Follow CLAUDE.md. Implement X."*

## What this is

Local-first multi-agent research intelligence platform. Given a research goal it plans,
searches literature, retrieves and parses papers, builds a RAG knowledge base, extracts
structured knowledge, verifies every claim against source text, synthesises, finds gaps,
critiques itself, and emits a report. Not a chatbot.

## Stack

Python 3.12 · FastAPI · LangGraph · LangChain · Ollama · Qdrant · Neo4j · PostgreSQL ·
Pydantic v2 · asyncio · Docker · uv · ruff · mypy · pytest.

## Layers (dependencies point inward only)

```
api  →  workflows  →  agents  →  services  →  integrations
                          ↘         ↓             ↑
                        schemas  repositories  core.interfaces
                                     ↓
                                   core
```

| Package | Owns | Must not |
|---|---|---|
| `core/` | settings, logging, exceptions, registry, events, retry, **interfaces (ports)** | import any other `researchagent` package |
| `config/` | YAML loading + typed config schemas | contain behaviour |
| `schemas/` | agent I/O contracts, workflow state | contain logic |
| `models/` | domain objects (Paper, Citation, ResearchGap) | know about storage or LLMs |
| `repositories/` | all DB access (Postgres, Qdrant, Neo4j) | be called from agents directly |
| `integrations/` | outbound adapters, one per external system | be imported by agents |
| `services/` | reusable capabilities agents compose | make control-flow decisions |
| `agents/` | reasoning, one responsibility each | perform I/O, or call another agent |
| `workflows/` | LangGraph orchestration, routing, retries | contain domain reasoning |
| `api/` | HTTP edge, error translation | contain business logic |

**Vendor SDKs may only be imported inside `integrations/`.** Everything else depends on a
port in `core/interfaces/`. Adding a provider = new adapter + one line of YAML.

## Non-negotiables

- Every agent subclasses `BaseAgent[TInput, TOutput]` and implements only `execute()`.
  Validation, retries, timing, logging and events are already handled by the base.
- Agents exchange Pydantic models. Never dicts.
- Orchestration lives in LangGraph, never inside an agent. Agents never call agents.
- No hardcoded models, temperatures, prompts, URLs, thresholds. They live in `config/*.yaml`
  and `prompts/`.
- Prompts are versioned files (`prompts/<agent>/v1.md`), never overwritten in place.
- `logging.get_logger(__name__)`, never `print`. Structured events, not f-strings.
- Errors: raise a `ResearchAgentError` subclass with context. Never swallow.
- Type hints everywhere; `mypy --strict` must pass.

## Adding an agent (the shape is always identical)

```
researchagent/agents/<name>/
    __init__.py
    agent.py      # BaseAgent subclass, @AGENTS.register("<name>")
    schemas.py    # <Name>Input / <Name>Output (+ *Draft schemas for the model)
    prompt.py     # message assembly from self.prompt (base resolves the version)
prompts/<name>/v1.md
tests/agents/<name>/test_agent.py
config/agents.yaml   # add the entry
```

## Workflow (target)

Planner → Discovery → Retrieval → Parser → Analysis → Verification → Synthesis →
Gap Discovery → Reviewer → (reject ⇒ back to Planner with feedback) → Report.

## Roadmap

v0.1 skeleton ✅ · v0.2 Planner + LangGraph ✅ · v0.3 Discovery + Retrieval ✅ ·
v0.4 Parsing · v0.5 RAG · v0.6 Verification · v0.7 Knowledge graph · v0.8 Reviewer +
memory · v0.9 UI · v1.0 evaluation.

Packages for later phases exist with a docstring stating their responsibility and are
empty until their version lands. That is intentional — do not fill them early.

## Commands

```bash
uv sync --extra dev          # install
uv run pytest                # test
uv run ruff check . && uv run ruff format .
uv run mypy researchagent    # strict
make dev                     # API with reload
docker compose up            # full local stack
```

## Working agreement

- One feature per request. Stop when it is done; do not continue into the next.
- Never redesign existing architecture without stating the conflict first.
- Reuse existing abstractions before adding new ones; no duplicate utilities.
- No placeholder or fake implementations. No invented APIs.
- Code first: ~90% implementation, ~10% explanation. No generated markdown unless asked.
- Finish with: files created, files modified, integration notes, commit message.
- Commits: `feat(planner): implement autonomous research planning agent`.
