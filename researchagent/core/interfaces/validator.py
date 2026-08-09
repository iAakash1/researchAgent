"""Validator port.

A validator answers one question about one kind of subject and returns a
:class:`ValidationResult` — never a bare bool, never an exception for an expected
negative. Validators are pure and synchronous: they reason over data already in hand, so
they are trivially testable and can be composed without an event loop.

One validator, one responsibility. Aggregate verdicts are produced by composing them,
not by writing a bigger validator.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import ClassVar

from researchagent.core.validation import ValidationResult


class Validator[T](ABC):
    """Validates subjects of type ``T``."""

    name: ClassVar[str]
    subject_type: ClassVar[str]

    @abstractmethod
    def check(self, subject: T) -> ValidationResult:
        """Inspect ``subject`` and report. Must not raise for an invalid subject."""

    def validate(self, subject: T) -> ValidationResult:
        """``check`` plus timing, so every result carries its own cost."""
        started = time.perf_counter()
        result = self.check(subject)
        elapsed_ms = (time.perf_counter() - started) * 1000
        return result.model_copy(update={"duration_ms": round(elapsed_ms, 3)})
