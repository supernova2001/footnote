from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from footnote.agents.judge import ApiJudge, CascadeJudge, JudgeOutput, LocalJudge, RecordingJudge, Ruling
from footnote.db.repo import MemoryRepo
from footnote.domain import Passage, Verdict
from footnote.judge_prompt import evidence_is_verbatim, parse_judge_output, render_judge_prompt
from footnote.llm.config import load_models_config
from footnote.llm.fake import FakeModels
from footnote.retrieval.retriever import reciprocal_rank_fusion
from footnote.retrieval.store import Hit
from footnote.services import Services
from footnote.settings import CONFIG_DIR

P = Passage(
    id="d:0001",
    document_id="d",
    title="T",
    section=None,
    scholar_ids=["s"],
    license_id="PD-US",
    source_url="https://example.org",
    text="Woodson launched Negro History Week in February 1926.",
)


def test_rrf_rewards_agreement_between_rankers() -> None:
    fused = reciprocal_rank_fusion([[Hit("a", 9), Hit("b", 8), Hit("c", 7)], [Hit("c", 0.9), Hit("a", 0.8)]])
    assert [h.id for h in fused][:2] == ["a", "c"]
    assert fused[0].score == pytest.approx(1 / 61 + 1 / 62)


@pytest.mark.parametrize("strategy", ["bm25", "dense", "hybrid", "hybrid_rerank"])
async def test_every_strategy_finds_the_passage(services: Services, strategy: str) -> None:
    passages = await services.retriever.search("When did Woodson launch Negro History Week?", strategy=strategy, k=3)  # type: ignore[arg-type]
    assert passages, strategy
    assert any("1926" in p.text for p in passages)
    assert all(p.license_id and p.source_url for p in passages)


async def test_scholar_and_document_filters(services: Services) -> None:
    only_wells = await services.retriever.search("lynching pamphlet", scholar_id="ida-b-wells", k=5)
    assert only_wells and all("ida-b-wells" in p.scholar_ids for p in only_wells)
    in_doc = await services.retriever.search("education", document_id="gutenberg-11089", k=5)
    assert in_doc and all(p.document_id == "gutenberg-11089" for p in in_doc)


def test_judge_output_parser_is_tolerant_of_fences_but_strict_on_contract() -> None:
    good = '```json\n{"label": "partial", "evidence": "x", "rationale": "y"}\n```'
    assert parse_judge_output(good) == {"label": "partial", "evidence": "x", "rationale": "y"}
    assert parse_judge_output('{"label": "maybe", "evidence": "", "rationale": ""}') is None
    assert parse_judge_output('{"label": "supported"}') is None
    assert parse_judge_output("not json") is None


def test_evidence_check_ignores_case_whitespace_and_curly_quotes() -> None:
    passages = [{"id": "1", "title": "t", "text": "He said “read—and   write”."}]
    assert evidence_is_verbatim('he said "read-and write"', passages)
    assert not evidence_is_verbatim("he said nothing", passages)
    assert not evidence_is_verbatim("", passages)


def test_prompt_is_the_contract() -> None:
    system, user = render_judge_prompt("A claim.", [P])
    assert (
        '"label"' in system
        and '<passage id="d:0001" title="T">' in user
        and user.endswith("<claim>\nA claim.\n</claim>")
    )


def _models(label: str, evidence: str) -> FakeModels:
    config = load_models_config(CONFIG_DIR / "models.yaml")
    return FakeModels(
        config, {"judge.grounding": lambda s, m, o: JudgeOutput(label=label, evidence=evidence, rationale="r")}
    )  # type: ignore[arg-type,misc]


async def test_claude_judge_fails_closed_on_hallucinated_evidence() -> None:
    ruling = await ApiJudge(_models("supported", "Woodson founded Harvard.")).rule("claim", [P])
    assert ruling.verdict.label == "unsupported" and "not found" in ruling.verdict.rationale
    ruling = await ApiJudge(_models("supported", "launched Negro History Week")).rule("claim", [P])
    assert ruling.verdict.label == "supported"


class Fixed:
    def __init__(self, name: str, label: str, valid: bool = True) -> None:
        self.name, self.label, self.valid, self.calls = name, label, valid, 0

    async def rule(self, claim: str, passages: Any) -> Ruling:
        self.calls += 1
        return Ruling(Verdict(label=self.label, evidence="", rationale="", judge=self.name), self.valid, 5)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("label", "valid", "escalates"),
    [
        ("supported", True, False),
        ("unsupported", True, False),
        ("partial", True, True),
        ("supported", False, True),
    ],
)
async def test_cascade_escalates_only_when_unsure(label: str, valid: bool, escalates: bool) -> None:
    local, claude = Fixed("local", label, valid), Fixed("claude", "supported")
    ruling = await CascadeJudge(local, claude, ["partial", "invalid"]).rule("c", [P])
    assert claude.calls == int(escalates) and ruling.escalated is escalates


async def test_recording_judge_writes_the_flywheel_row() -> None:
    repo = MemoryRepo()
    await RecordingJudge(Fixed("local", "supported"), repo).rule("c", [P])
    row = repo.judgments[0]
    assert row["prompt_version"] == "judge.v2" and row["passages"][0]["id"] == P.id and row["human_label"] is None


async def test_local_judge_speaks_openai_compatible_json() -> None:
    def reply(content: str) -> httpx.MockTransport:
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            assert body["response_format"]["type"] == "json_schema" and body["temperature"] == 0
            return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

        return httpx.MockTransport(handler)

    ok = LocalJudge(
        "http://x/v1",
        "footnote-judge",
        client=httpx.AsyncClient(
            transport=reply('{"label": "supported", "evidence": "in February 1926", "rationale": "stated"}')
        ),
    )
    assert (await ok.rule("c", [P])).verdict.label == "supported"
    bad = LocalJudge("http://x/v1", "footnote-judge", client=httpx.AsyncClient(transport=reply("I think yes")))
    ruling = await bad.rule("c", [P])
    assert not ruling.valid and ruling.verdict.label == "unsupported"


def test_elided_quotes_are_verbatim_only_when_every_fragment_is() -> None:
    passages = [
        {
            "id": "1",
            "title": "t",
            "text": "Wells gathered newspaper reports and testimony to show that the reasons were false.",
        }
    ]
    assert evidence_is_verbatim("Wells gathered newspaper reports ... the reasons were false", passages)
    assert evidence_is_verbatim("Wells gathered newspaper reports … the reasons were false", passages)
    assert not evidence_is_verbatim("Wells gathered newspaper reports ... the reasons were true", passages)


def test_judge_prompt_names_each_passage_source() -> None:
    p = P.model_copy(update={"attribution": 'Ida B. Wells, "Southern Horrors" (1892), public domain'})
    _, user = render_judge_prompt("Wells argued X.", [p])
    assert "source=\"Ida B. Wells, 'Southern Horrors' (1892), public domain\"" in user
    _, bare = render_judge_prompt("c", [{"id": "1", "title": "t", "text": "x"}])
    assert "source=" not in bare
