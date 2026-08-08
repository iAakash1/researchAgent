from __future__ import annotations

import pytest

from researchagent.core.exceptions import RegistryError
from researchagent.core.registry import Registry


def test_register_and_get_is_case_insensitive() -> None:
    registry: Registry[str] = Registry("thing")
    registry.add("Ollama", "value")

    assert registry.get("ollama") == "value"
    assert registry.get("  OLLAMA  ") == "value"
    assert "ollama" in registry


def test_duplicate_registration_is_rejected() -> None:
    registry: Registry[str] = Registry("thing")
    registry.add("a", "1")

    with pytest.raises(RegistryError):
        registry.add("a", "2")

    registry.add("a", "2", override=True)
    assert registry.get("a") == "2"


def test_unknown_key_lists_available_options() -> None:
    registry: Registry[str] = Registry("provider")
    registry.add("ollama", "x")

    with pytest.raises(RegistryError) as excinfo:
        registry.get("openai")

    assert excinfo.value.context["available"] == ["ollama"]
    assert excinfo.value.context["kind"] == "provider"


def test_empty_name_is_rejected() -> None:
    registry: Registry[str] = Registry("thing")
    with pytest.raises(RegistryError):
        registry.add("   ", "x")


def test_decorator_registration() -> None:
    registry: Registry[type] = Registry("agent")

    @registry.register("planner")
    class Planner:
        pass

    assert registry.get("planner") is Planner
    assert registry.names() == ("planner",)
    assert len(registry) == 1
