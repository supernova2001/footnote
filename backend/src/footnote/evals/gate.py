"""Deterministic release gate for a completed production evaluation report."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any


def validate_report(report: dict[str, Any], policy: dict[str, Any], now: datetime | None = None) -> list[str]:
    errors: list[str] = []
    if report.get("mode") != "production":
        errors.append("report mode is not production")
    if report.get("run_scope") != "all":
        errors.append("report was not produced by a complete `eval all` run")
    provenance = report.get("provenance") or {}
    if not provenance.get("answer_dataset", {}).get("reviewed"):
        errors.append("answer dataset is not recorded as human-reviewed")
    if not provenance.get("judge_dataset", {}).get("human_labeled"):
        errors.append("judge dataset is not recorded as human-labeled")

    generated = report.get("generated_at")
    try:
        timestamp = datetime.fromisoformat(str(generated).replace("Z", "+00:00"))
        age_hours = ((now or datetime.now(UTC)) - timestamp.astimezone(UTC)).total_seconds() / 3600
        if age_hours < 0 or age_hours > float(policy["max_age_hours"]):
            errors.append(f"report age {age_hours:.1f}h exceeds {policy['max_age_hours']}h")
    except (TypeError, ValueError):
        errors.append("generated_at is missing or invalid")

    for suite, requirement in policy.get("suites", {}).items():
        rows = report.get(suite)
        if not isinstance(rows, list):
            errors.append(f"{suite}: result rows are missing")
            continue
        selector = requirement.get("selector", {})
        matched = [row for row in rows if all(row.get(key) == value for key, value in selector.items())]
        if len(matched) != 1:
            errors.append(f"{suite}: expected exactly one row matching {selector}, found {len(matched)}")
            continue
        row = matched[0]
        if int(row.get("n", 0)) < int(requirement["min_cases"]):
            errors.append(f"{suite}: n={row.get('n', 0)} is below {requirement['min_cases']}")
        for metric, minimum in requirement.get("minimums", {}).items():
            value = row.get(metric)
            if not isinstance(value, int | float) or value < minimum:
                errors.append(f"{suite}: {metric}={value!r} is below {minimum}")
    return errors
