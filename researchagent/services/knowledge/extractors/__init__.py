"""Specialized extractors — one kind of knowledge each, no giant extractor.

Splitting by kind is what makes the prompts specific, the section selection meaningful,
and a failure attributable: when results are wrong, exactly one prompt and one mapper are
implicated.
"""

from researchagent.services.knowledge.extractors.dataset import DatasetExtractor
from researchagent.services.knowledge.extractors.future_work import FutureWorkExtractor
from researchagent.services.knowledge.extractors.limitation import LimitationExtractor
from researchagent.services.knowledge.extractors.method import MethodExtractor
from researchagent.services.knowledge.extractors.metric import MetricExtractor
from researchagent.services.knowledge.extractors.result import ResultExtractor

__all__ = [
    "DatasetExtractor",
    "FutureWorkExtractor",
    "LimitationExtractor",
    "MethodExtractor",
    "MetricExtractor",
    "ResultExtractor",
]
