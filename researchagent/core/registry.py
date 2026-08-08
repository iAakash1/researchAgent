"""Generic name -> object registry.

Used for LLM providers, agents and tools so that new components are discovered by
name from YAML config instead of being imported and wired by hand.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

from researchagent.core.exceptions import RegistryError


class Registry[T]:
    """A small, explicit registry keyed by lowercase names."""

    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._items: dict[str, T] = {}

    @property
    def kind(self) -> str:
        return self._kind

    def add(self, name: str, item: T, *, override: bool = False) -> T:
        key = self._normalise(name)
        if key in self._items and not override:
            raise RegistryError(f"Duplicate {self._kind} registration", name=key, kind=self._kind)
        self._items[key] = item
        return item

    def register(self, name: str, *, override: bool = False) -> Callable[[T], T]:
        """Decorator form: ``@registry.register("ollama")``."""

        def decorator(item: T) -> T:
            self.add(name, item, override=override)
            return item

        return decorator

    def get(self, name: str) -> T:
        key = self._normalise(name)
        try:
            return self._items[key]
        except KeyError:
            raise RegistryError(
                f"Unknown {self._kind}",
                name=key,
                kind=self._kind,
                available=sorted(self._items),
            ) from None

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._items))

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and self._normalise(name) in self._items

    def __iter__(self) -> Iterator[tuple[str, T]]:
        return iter(sorted(self._items.items()))

    def __len__(self) -> int:
        return len(self._items)

    @staticmethod
    def _normalise(name: str) -> str:
        key = name.strip().lower()
        if not key:
            raise RegistryError("Registry name must not be empty")
        return key
