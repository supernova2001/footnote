"""The training rows use the backend's prompt, byte for byte, and say which version."""

from __future__ import annotations

import importlib.util
import json

import pytest

from footnote_ft import shared
from footnote_ft.formatting import check_prompt_version, system_user, target_text


def test_shared_module_is_the_backend_file():
    folder = shared.shared_dir()
    assert folder.parts[-3:] == ("backend", "src", "footnote")
    assert (folder / "judge_prompt.py").is_file()
    # Loaded independently from the same path, it renders the same prompt.
    spec = importlib.util.spec_from_file_location("judge_prompt_direct", folder / "judge_prompt.py")
    direct = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(direct)
    passages = [{"id": "x:0000", "title": "T", "text": "Some text."}]
    assert direct.render_judge_prompt("A claim.", passages) == shared.judge_prompt().render_judge_prompt(
        "A claim.", passages
    )
    assert direct.PROMPT_VERSION == shared.judge_prompt().PROMPT_VERSION


def test_every_row_uses_render_judge_prompt(rows):
    jp = shared.judge_prompt()
    assert rows
    for row in rows:
        system, user = jp.render_judge_prompt(row["claim"], row["passages"])
        assert row["messages"][0] == {"role": "system", "content": system}
        assert row["messages"][1] == {"role": "user", "content": user}
        assert row["prompt_version"] == jp.PROMPT_VERSION
        assert system_user(row) == (system, user)


def test_assistant_target_meets_the_contract(rows):
    jp = shared.judge_prompt()
    for row in rows:
        target = target_text(row)
        parsed = jp.parse_judge_output(target)
        assert parsed is not None, target
        assert list(json.loads(target)) == ["label", "evidence", "rationale"]
        assert parsed["label"] == row["label"]
        if row["label"] == "unsupported":
            assert parsed["evidence"] == ""
        else:
            assert jp.evidence_is_verbatim(parsed["evidence"], row["passages"])
        assert parsed["rationale"].strip()


def test_stale_prompt_version_is_refused(rows):
    stale = {**rows[0], "prompt_version": "judge.v0"}
    with pytest.raises(ValueError, match="rebuild the dataset"):
        check_prompt_version(stale)
