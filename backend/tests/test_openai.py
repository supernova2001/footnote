from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest

from footnote.agents.mcp_tools import open_tools
from footnote.agents.openai_agents import OpenAIResearchAgent, run_tool_loop
from footnote.agents.tool_agents import ResearchResult, SourcePick
from footnote.llm.client import search_result_block
from footnote.llm.config import load_models_config
from footnote.llm.openai_client import parse_cited_text, to_openai_input
from footnote.llm.router import RoutedModels
from footnote.services import Services
from footnote.settings import CONFIG_DIR


def test_search_results_are_numbered_and_markers_map_back_to_passage_ids() -> None:
    messages = [
        {
            "role": "user",
            "content": [
                search_result_block(
                    "doc:0001", "Title A", ["Woodson founded the journal in 1916.", "It was scholarly."]
                ),
                search_result_block("doc:0002", "Title B", ["Negro History Week began in 1926."]),
                {"type": "text", "text": "Question: when?"},
            ],
        }
    ]
    inp, numbered = to_openai_input(messages)
    assert numbered[1]["source"] == "doc:0001" and numbered[2]["source"] == "doc:0002"
    assert "[1] Title A" in inp[0]["content"] and inp[0]["content"].endswith("Question: when?")

    text = "Woodson founded a journal in 1916 [1]. The week began in 1926 [2][7]. It grew."
    blocks = parse_cited_text(text, numbered)
    assert [b.text for b in blocks] == ["Woodson founded a journal in 1916.", " The week began in 1926.", " It grew."]
    assert [c.source for c in blocks[0].citations] == ["doc:0001"]
    assert [c.source for c in blocks[1].citations] == ["doc:0002"]  # [7] names nothing shown: dropped
    assert blocks[0].citations[0].cited_text == "Woodson founded the journal in 1916."
    assert blocks[2].citations == []


def test_marker_styles() -> None:
    numbered = {1: {"source": "a", "sentences": ["x"]}, 2: {"source": "b", "sentences": ["y"]}}
    assert [c.source for c in parse_cited_text("One. [1, 2]", numbered)[0].citations] == ["a", "b"]
    assert parse_cited_text("No citations here.", numbered)[0].citations == []


@pytest.mark.parametrize("preset", ["openai", "openai-quality"])
def test_openai_preset_switches_every_route_and_keeps_judge_independent(preset: str) -> None:
    config = load_models_config(CONFIG_DIR / "models.yaml", preset)
    assert config.providers() == {"openai"}
    assert config.route("judge.grounding").model != config.route("ask.answer").model
    assert config.route("evals.grader").model not in {
        config.route("judge.grounding").model,
        config.route("ask.answer").model,
    }
    assert all(config.pricing.get(r.model) for r in config.routes.values())  # every model is priced
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        RoutedModels(config, {})


# ── the OpenAI tool loop, against the real in-process MCP server ─────────────────────────


@dataclass
class ScriptedResponses:
    """Stands in for AsyncOpenAI().responses: turn 1 calls search_passages, turn 2 answers."""

    calls: list[dict[str, Any]] = field(default_factory=list)

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        usage = SimpleNamespace(input_tokens=100, output_tokens=20)
        if len(self.calls) == 1:
            call = SimpleNamespace(
                type="function_call",
                name="search_passages",
                call_id="c1",
                arguments='{"query": "Negro History Week", "k": 2}',
            )
            return SimpleNamespace(
                id="r1", output=[call], output_text="", usage=usage, model="gpt-test", status="completed"
            )
        return SimpleNamespace(
            id="r2", output=[], output_text="Done searching.", usage=usage, model="gpt-test", status="completed"
        )

    async def parse(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        hit = self.calls[1]["input"][0]["output"]
        import json

        first_id = json.loads(hit)["result"][0]["id"]
        parsed = ResearchResult(picks=[SourcePick(passage_id=first_id, reason="covers it")], not_covered="")
        return SimpleNamespace(
            id="r3",
            output=[],
            output_text="",
            output_parsed=parsed,
            model="gpt-test",
            status="completed",
            usage=SimpleNamespace(input_tokens=50, output_tokens=10),
        )


async def test_tool_loop_runs_mcp_tools_and_returns_structured_picks(services: Services) -> None:
    config = load_models_config(CONFIG_DIR / "models.yaml", "openai")
    responses = ScriptedResponses()
    client: Any = SimpleNamespace(responses=responses)
    agent = OpenAIResearchAgent(client, config, services.repo, services.mcp_server)
    passages = await agent.find_sources("Carter G. Woodson and Negro History Week", None)
    assert passages and passages[0].license_id
    create_1, create_2, parse_3 = responses.calls
    assert {t["name"] for t in create_1["tools"]} >= {"search_passages", "get_passage"}
    assert "ask_corpus" not in {t["name"] for t in create_1["tools"]}  # research gets read-only tools
    assert create_2["previous_response_id"] == "r1" and create_2["input"][0]["call_id"] == "c1"
    assert parse_3["text_format"] is ResearchResult and parse_3["previous_response_id"] == "r2"


async def test_tool_loop_reports_unknown_tools_to_the_model(services: Services) -> None:
    config = load_models_config(CONFIG_DIR / "models.yaml", "openai")

    class Unknown(ScriptedResponses):
        async def create(self, **kwargs: Any) -> Any:
            self.calls.append(kwargs)
            usage = SimpleNamespace(input_tokens=1, output_tokens=1)
            if len(self.calls) == 1:
                call = SimpleNamespace(type="function_call", name="drop_tables", call_id="x", arguments="{}")
                return SimpleNamespace(
                    id="r1", output=[call], output_text="", usage=usage, model="m", status="completed"
                )
            return SimpleNamespace(id="r2", output=[], output_text="ok", usage=usage, model="m", status="completed")

    responses = Unknown()
    from contextlib import AsyncExitStack

    async with AsyncExitStack() as stack:
        pairs = await open_tools(stack, services.mcp_server(), allowed=["search_passages"])
        text, _ = await run_tool_loop(
            SimpleNamespace(responses=responses),
            config,
            "chat.supervisor",  # type: ignore[arg-type]
            system="s",
            messages=[{"role": "user", "content": "hi"}],
            pairs=pairs,
        )
    assert text == "ok"
    assert responses.calls[1]["input"][0]["output"].startswith("Error: no tool named")


def test_default_openai_preset_is_the_cheap_one() -> None:
    cheap = load_models_config(CONFIG_DIR / "models.yaml", "openai")
    quality = load_models_config(CONFIG_DIR / "models.yaml", "openai-quality")

    def cost(c):  # the measured live token profile of one answer: writer-side vs judge-side
        writer = c.usage(c.route("ask.answer").model, 2_100, 350).cost_usd
        judge = c.usage(c.route("judge.grounding").model, 2_900, 450).cost_usd
        return writer + judge

    assert cost(cheap) * 10 < cost(quality)
