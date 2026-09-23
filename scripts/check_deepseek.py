"""One tiny live DeepSeek call for manual credential and model verification."""

from __future__ import annotations

import asyncio

from researchagent.config.loader import ConfigLoader
from researchagent.config.schemas import ModelCatalog
from researchagent.core.interfaces.llm import GenerationParams, Message
from researchagent.core.settings import get_settings
from researchagent.integrations.registry import build_llm_provider


async def main() -> None:
    settings = get_settings()
    catalog = ConfigLoader(settings.config_dir).load("models", ModelCatalog)
    spec = catalog.spec_for("fast")
    provider = build_llm_provider(spec.provider, settings)
    try:
        response = await provider.complete(
            [Message.user("Reply with exactly: OK")],
            model=spec.model_name,
            params=spec.params.merged_with(GenerationParams(thinking=False, max_output_tokens=16)),
        )
        print(
            f"provider={response.provider} model={response.model} "
            f"tokens={response.usage.total_tokens} response={response.text.strip()}"
        )
    finally:
        await provider.aclose()


if __name__ == "__main__":
    asyncio.run(main())
