from __future__ import annotations

from datetime import UTC, datetime, timedelta

from footnote.evals.gate import validate_report

POLICY = {
    "max_age_hours": 24,
    "suites": {
        "answers": {
            "selector": {"strategy": "hybrid_rerank"},
            "min_cases": 50,
            "minimums": {"faithfulness": 0.95},
        }
    },
}


def report(now: datetime) -> dict:
    return {
        "mode": "production",
        "run_scope": "all",
        "generated_at": now.isoformat(),
        "provenance": {
            "answer_dataset": {"reviewed": True},
            "judge_dataset": {"human_labeled": True},
        },
        "answers": [{"strategy": "hybrid_rerank", "n": 50, "faithfulness": 0.96}],
    }


def test_production_eval_report_passes_release_gate() -> None:
    now = datetime(2026, 11, 20, tzinfo=UTC)
    assert validate_report(report(now), POLICY, now) == []


def test_eval_gate_rejects_stale_unreviewed_or_weak_results() -> None:
    now = datetime(2026, 11, 20, tzinfo=UTC)
    value = report(now - timedelta(hours=25))
    value["mode"] = "offline"
    value["provenance"]["answer_dataset"]["reviewed"] = False
    value["answers"][0].update(n=10, faithfulness=0.7)
    errors = validate_report(value, POLICY, now)
    assert any("not production" in error for error in errors)
    assert any("human-reviewed" in error for error in errors)
    assert any("report age" in error for error in errors)
    assert any("n=10" in error for error in errors)
    assert any("faithfulness" in error for error in errors)
