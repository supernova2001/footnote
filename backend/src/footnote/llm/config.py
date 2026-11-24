"""Model routes, judge backend, limits and prices, loaded from config/models.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

from footnote.domain import Usage

Effort = Literal["low", "medium", "high", "xhigh", "max"]


Provider = Literal["anthropic", "openai"]


class RouteConfig(BaseModel):
    provider: Provider = "anthropic"
    model: str
    effort: Effort = "medium"
    max_tokens: int = 8000


class JudgeConfig(BaseModel):
    backend: Literal["api", "local", "cascade"] = "api"
    escalate_on: list[Literal["partial", "invalid", "unsupported"]] = ["partial", "invalid"]
    concurrency: int = 6


class AskLimits(BaseModel):
    max_subqueries: int = 3
    retrieve_k: int = 24
    rerank_top_n: int = 8
    max_revisions: int = 1


class WriterLimits(BaseModel):
    max_sections: int = 5
    section_retries: int = 2
    editor_retries: int = 1
    passages_per_section: int = 6


class AgentLimits(BaseModel):
    max_iterations: int = 8


class Limits(BaseModel):
    ask: AskLimits = AskLimits()
    writer: WriterLimits = WriterLimits()
    agent: AgentLimits = AgentLimits()


class Price(BaseModel):
    input: float
    output: float


class ModelsConfig(BaseModel):
    routes: dict[str, RouteConfig]
    presets: dict[str, dict[str, RouteConfig]] = Field(default_factory=dict)
    judge: JudgeConfig = JudgeConfig()
    limits: Limits = Limits()
    pricing: dict[str, Price] = Field(default_factory=dict)

    def with_preset(self, name: str | None) -> ModelsConfig:
        """Every route re-pointed at a provider's preset (e.g. FOOTNOTE_LLM_PROVIDER=openai)."""
        if not name or (name == "anthropic" and "anthropic" not in self.presets):
            return self
        if name not in self.presets:
            raise KeyError(f"No preset {name!r} in models.yaml; known: {sorted(self.presets)}")
        preset = self.presets[name]
        missing = sorted(set(self.routes) - set(preset))
        if missing:
            raise KeyError(f"Preset {name!r} has no model for routes {missing}")
        return self.model_copy(update={"routes": {**self.routes, **preset}})

    def providers(self) -> set[str]:
        return {r.provider for r in self.routes.values()}

    def route(self, name: str) -> RouteConfig:
        try:
            return self.routes[name]
        except KeyError:
            raise KeyError(f"No model route {name!r} in models.yaml; known: {sorted(self.routes)}") from None

    def usage(self, model: str, input_tokens: int, output_tokens: int) -> Usage:
        price = self.pricing.get(model) or self.pricing.get(model.split(":", 1)[0])
        cost = 0.0
        if price:
            cost = (input_tokens * price.input + output_tokens * price.output) / 1_000_000
        return Usage(input_tokens=input_tokens, output_tokens=output_tokens, cost_usd=round(cost, 6))


def load_models_config(path: Path, preset: str | None = None) -> ModelsConfig:
    with path.open(encoding="utf-8") as f:
        return ModelsConfig.model_validate(yaml.safe_load(f)).with_preset(preset)
