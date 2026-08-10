class TestNestedOverrides:
    """Layering must produce models, not dicts.

    `model_copy(update=...)` does not validate, so an agent overriding a nested block used
    to receive a plain dict — which only failed later, inside the retry helper.
    """

    def test_a_per_agent_retry_policy_stays_a_retry_policy(self) -> None:
        from researchagent.config.schemas import AgentConfig
        from researchagent.core.retry import RetryPolicy

        config = AgentConfig.model_validate(
            {
                "defaults": {"model": "reasoning", "retry": {"max_attempts": 3}},
                "agents": {"verification": {"retry": {"max_attempts": 2}}},
            }
        )

        spec = config.spec_for("verification")

        assert isinstance(spec.retry, RetryPolicy)
        assert spec.retry.max_attempts == 2

    def test_unset_nested_fields_still_come_from_defaults(self) -> None:
        from researchagent.config.schemas import AgentConfig

        config = AgentConfig.model_validate(
            {
                "defaults": {"model": "reasoning", "timeout_seconds": 300},
                "agents": {"reviewer": {"prompt_version": "v1"}},
            }
        )

        assert config.spec_for("reviewer").timeout_seconds == 300
