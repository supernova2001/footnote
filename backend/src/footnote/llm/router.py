"""One `Models` for the graphs, dispatching each route to the provider models.yaml names for it."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any, TypeVar

from pydantic import BaseModel

from footnote.llm.client import AnswerEvent, Models
from footnote.llm.config import ModelsConfig

T = TypeVar("T", bound=BaseModel)


class RoutedModels:
    def __init__(self, config: ModelsConfig, providers: Mapping[str, Models]) -> None:
        self.config = config
        self.providers = dict(providers)
        missing = config.providers() - set(self.providers)
        if missing:
            raise RuntimeError(
                f"models.yaml routes use {sorted(missing)} but no key is configured for it "
                "(ANTHROPIC_API_KEY / OPENAI_API_KEY), or set FOOTNOTE_LLM_PROVIDER to a provider you have"
            )

    def _for(self, route: str) -> Models:
        return self.providers[self.config.route(route).provider]

    def model_for(self, route: str) -> str:
        return self.config.route(route).model

    async def structured(
        self,
        route: str,
        *,
        system: str,
        messages: Sequence[dict[str, Any]],
        output: type[T],
        cache_system: bool = False,
    ) -> T:
        return await self._for(route).structured(
            route, system=system, messages=messages, output=output, cache_system=cache_system
        )

    def stream_answer(
        self, route: str, *, system: str, messages: Sequence[dict[str, Any]]
    ) -> AsyncIterator[AnswerEvent]:
        return self._for(route).stream_answer(route, system=system, messages=messages)
