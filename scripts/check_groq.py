"""Opt-in live check against the real Groq API.

Deliberately NOT a pytest test: `uv run pytest` must pass offline, with no key and no
account. Run this only when you want to confirm that the configured model id still exists
and that structured output still behaves:

    uv run python scripts/check_groq.py

Reads GROQ_API_KEY from the environment (or .env) and exits non-zero with a plain message
if it is absent. Prints no part of the key.
"""

from __future__ import annotations

import asyncio
import sys

from pydantic import BaseModel

from researchagent.config import ConfigLoader, ModelCatalog
from researchagent.core.exceptions import ResearchAgentError
from researchagent.core.interfaces.llm import Message
from researchagent.core.settings import get_settings
from researchagent.integrations.registry import build_llm_provider


class Answer(BaseModel):
    """Small schema, chosen so a wrong answer is as visible as a malformed one."""

    capital: str
    confidence: float


async def main() -> int:
    settings = get_settings()
    if settings.groq_api_key is None:
        print("GROQ_API_KEY is not set — skipping. This script is opt-in by design.")
        return 1

    catalog = ConfigLoader(settings.config_dir).load("models", ModelCatalog)
    aliases = [alias for alias, spec in catalog.models.items() if spec.provider == "groq"]
    if not aliases:
        print("No model alias in config/models.yaml uses provider: groq.")
        return 1

    provider = build_llm_provider("groq", settings)
    try:
        for alias in aliases:
            spec = catalog.spec_for(alias)
            print(f"\n[{alias}] {spec.model_name}")

            health = await provider.health()
            served = spec.model_name in health.available_models
            print(f"  reachable: {health.healthy}   model served: {served}")
            if not served:
                print("  -> Groq no longer serves this id. Update config/models.yaml.")
                continue

            response = await provider.complete(
                [Message.user("Reply with one word: the capital of France.")],
                model=spec.model_name,
                params=spec.params,
            )
            print(f"  completion: {response.text.strip()[:60]!r}")
            print(f"  tokens: {response.usage.total_tokens}  latency: {response.latency_ms:.0f}ms")

            structured = await provider.complete_structured(
                [Message.user("What is the capital of France?")],
                model=spec.model_name,
                params=spec.params,
                schema=Answer,
            )
            print(f"  structured: {structured.model_dump()}")
    except ResearchAgentError as error:
        # to_dict() carries code/remedy/context and never the credential.
        print(f"\nFAILED: {error.to_dict()}")
        return 1
    finally:
        await provider.aclose()

    print("\nOK")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
