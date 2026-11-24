"""The shared judge metrics (backend/src/footnote/judge_metrics.py), loaded the way eval_judge.py loads them."""

from __future__ import annotations

import json

import pytest

from footnote_ft import shared

m = shared.judge_metrics()
jp = shared.judge_prompt()
PASSAGES = [{"id": "d:0000", "title": "T", "text": "The clinic was opened in 1897 with two beds."}]


def reply(label: str, evidence: str = "", rationale: str = "r") -> str:
    return json.dumps({"label": label, "evidence": evidence, "rationale": rationale})


def test_parse_tolerates_fences_but_not_broken_contracts():
    assert jp.parse_judge_output("```json\n" + reply("partial", "x") + "\n```")["label"] == "partial"
    assert jp.parse_judge_output('{"label": "maybe", "evidence": "", "rationale": ""}') is None
    assert jp.parse_judge_output('{"label": "supported", "rationale": ""}') is None
    assert jp.parse_judge_output("Supported.") is None


def test_scores_match_a_hand_computed_confusion():
    ev = "The clinic was opened in 1897"
    cases = [
        m.JudgeCase("supported", reply("supported", ev), PASSAGES, 100),
        m.JudgeCase("supported", reply("partial", ev), PASSAGES, 200),
        m.JudgeCase("partial", reply("partial", ev), PASSAGES, 300),
        m.JudgeCase("partial", "not json", PASSAGES, 400),
        m.JudgeCase("unsupported", reply("unsupported"), PASSAGES, 500),
        m.JudgeCase("unsupported", reply("supported", "made-up quote"), PASSAGES, 600),
    ]
    r = m.score_judge(cases)
    # supported: P=1/2 R=1/2 F1=.5 | partial: P=1/2 R=1/2 F1=.5 | unsupported: P=1 R=1/2 F1=2/3
    assert r.per_label["supported"].f1 == pytest.approx(0.5)
    assert r.per_label["partial"].f1 == pytest.approx(0.5)
    assert r.per_label["unsupported"].f1 == pytest.approx(2 / 3)
    assert r.macro_f1 == pytest.approx((0.5 + 0.5 + 2 / 3) / 3)
    assert r.accuracy == pytest.approx(3 / 6)
    assert r.unsupported_recall == pytest.approx(0.5)
    assert r.json_valid_rate == pytest.approx(5 / 6)
    assert r.confusion["partial"]["invalid"] == 1
    # 4 valid non-unsupported replies, 3 with a real quote
    assert r.evidence_verbatim_rate == pytest.approx(3 / 4)
    # deployed: the invalid reply and the made-up quote both become "unsupported"
    assert r.deployed_unsupported_recall == pytest.approx(1.0)
    assert r.p50_latency_ms == pytest.approx(350)
    assert r.p95_latency_ms == pytest.approx(575)


def test_eval_row_has_exactly_the_contract_keys():
    r = m.score_judge([m.JudgeCase("unsupported", reply("unsupported"), PASSAGES, 10)])
    row = m.eval_row(r, "local:footnote-judge", 0.0)
    assert list(row) == [
        "judge",
        "n",
        "macro_f1",
        "unsupported_recall",
        "json_valid_rate",
        "p50_latency_ms",
        "cost_per_1k_usd",
    ]


def test_bad_inputs_are_refused():
    with pytest.raises(ValueError):
        m.score_judge([])
    with pytest.raises(ValueError):
        m.score_judge([m.JudgeCase("maybe", reply("supported"))])


def test_percentile_matches_numpy_linear():
    assert m.percentile([1, 2, 3, 4], 50) == pytest.approx(2.5)
    assert m.percentile([5], 95) == 5
    assert m.percentile([], 50) is None
