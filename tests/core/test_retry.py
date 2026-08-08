from __future__ import annotations

import pytest

from researchagent.core.exceptions import ConfigurationError, OutputParsingError
from researchagent.core.retry import RetryPolicy, retry_async

FAST = RetryPolicy(max_attempts=3, initial_delay_seconds=0.0, max_delay_seconds=0.0, jitter=False)


async def test_returns_immediately_on_success() -> None:
    calls = 0

    async def operation() -> str:
        nonlocal calls
        calls += 1
        return "ok"

    result, attempts = await retry_async(operation, FAST, operation_name="test")

    assert (result, attempts, calls) == ("ok", 1, 1)


async def test_retries_retryable_errors_until_success() -> None:
    calls = 0

    async def operation() -> str:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise OutputParsingError("bad json")
        return "ok"

    result, attempts = await retry_async(operation, FAST, operation_name="test")

    assert (result, attempts) == ("ok", 3)


async def test_non_retryable_error_fails_on_first_attempt() -> None:
    calls = 0

    async def operation() -> str:
        nonlocal calls
        calls += 1
        raise ConfigurationError("missing file")

    with pytest.raises(ConfigurationError):
        await retry_async(operation, FAST, operation_name="test")

    assert calls == 1


async def test_raises_last_error_when_attempts_exhausted() -> None:
    async def operation() -> str:
        raise OutputParsingError("still bad")

    with pytest.raises(OutputParsingError):
        await retry_async(operation, FAST, operation_name="test")


async def test_on_retry_callback_receives_each_failure() -> None:
    seen: list[int] = []

    async def on_retry(attempt: int, error: BaseException) -> None:
        seen.append(attempt)

    async def operation() -> str:
        raise OutputParsingError("bad")

    with pytest.raises(OutputParsingError):
        await retry_async(operation, FAST, operation_name="test", on_retry=on_retry)

    # Fires after failed attempts 1 and 2, not after the terminal attempt 3.
    assert seen == [1, 2]


def test_backoff_grows_and_is_capped() -> None:
    policy = RetryPolicy(
        initial_delay_seconds=1.0,
        max_delay_seconds=4.0,
        backoff_multiplier=2.0,
        jitter=False,
        max_attempts=6,
    )

    assert [policy.delay_for(n) for n in (2, 3, 4, 5)] == [1.0, 2.0, 4.0, 4.0]
