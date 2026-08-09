"""Liveness and readiness endpoints.

``/health/live``  — the process is up (no dependencies touched).
``/health/ready`` — every dependency this version actually uses is usable.

Dependencies are added to the readiness probe as their subsystems land, so a green
``/health/ready`` always means "the features that exist will work".
"""

from __future__ import annotations

from fastapi import APIRouter, Response, status
from pydantic import BaseModel, Field

from researchagent.api.dependencies import LLMServiceDep, SettingsDep
from researchagent.core.constants import APP_NAME, APP_VERSION
from researchagent.core.interfaces.llm import ProviderHealth

router = APIRouter(prefix="/health", tags=["health"])


class LivenessResponse(BaseModel):
    status: str = "ok"
    service: str = APP_NAME
    version: str = APP_VERSION
    environment: str


class ModelAvailability(BaseModel):
    alias: str
    model: str
    provider: str
    pulled: bool


class ReadinessResponse(BaseModel):
    ready: bool
    providers: list[ProviderHealth] = Field(default_factory=list)
    models: list[ModelAvailability] = Field(default_factory=list)
    # Optional providers named in the catalogue but lacking credentials here.
    # Listed rather than omitted so "not configured" is visibly different
    # from "working".
    unconfigured_providers: list[str] = Field(default_factory=list)


@router.get("/live", response_model=LivenessResponse)
async def liveness(settings: SettingsDep) -> LivenessResponse:
    return LivenessResponse(environment=settings.environment.value)


@router.get("/ready", response_model=ReadinessResponse)
async def readiness(llm_service: LLMServiceDep, response: Response) -> ReadinessResponse:
    providers = await llm_service.health()
    availability = await llm_service.verify_models_available()

    models = [
        ModelAvailability(
            alias=alias,
            model=spec.model_name,
            provider=spec.provider,
            pulled=availability.get(alias, False),
        )
        for alias, spec in sorted(llm_service.active_aliases().items())
    ]
    _, unconfigured = llm_service.configured_providers()

    ready = all(p.healthy for p in providers) and all(m.pulled for m in models)
    if not ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return ReadinessResponse(
        ready=ready,
        providers=providers,
        models=models,
        unconfigured_providers=sorted(unconfigured),
    )
