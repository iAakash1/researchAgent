"""Extractor registry.

``config/knowledge.yaml`` names extractors as strings; this turns those names into
instances. Adding a knowledge kind is a new extractor, a new prompt, and one line of
YAML — no change to the pipeline, the validators, or the workflow.
"""

from __future__ import annotations

from typing import Any

from researchagent.core.logging import get_logger
from researchagent.core.prompts import PromptLibrary
from researchagent.core.registry import Registry
from researchagent.services.knowledge.base import KnowledgeExtractor
from researchagent.services.knowledge.extractors import (
    DatasetExtractor,
    FutureWorkExtractor,
    LimitationExtractor,
    MethodExtractor,
    MetricExtractor,
    ResultExtractor,
)
from researchagent.services.llm_service import BoundLLM

logger = get_logger(__name__)

EXTRACTORS: Registry[type[KnowledgeExtractor[Any, Any]]] = Registry("knowledge_extractor")

for extractor_class in (
    MethodExtractor,
    DatasetExtractor,
    MetricExtractor,
    ResultExtractor,
    LimitationExtractor,
    FutureWorkExtractor,
):
    EXTRACTORS.add(extractor_class.name, extractor_class)


def build_extractors(
    names: tuple[str, ...], llm: BoundLLM, prompts: PromptLibrary, *, prompt_version: str = "v1"
) -> list[KnowledgeExtractor[Any, Any]]:
    """Instantiate the extractors enabled in configuration."""
    extractors = [
        EXTRACTORS.get(name)(llm, prompts, prompt_version=prompt_version) for name in names
    ]
    logger.info("knowledge_extractors_built", enabled=[e.name for e in extractors])
    return extractors
