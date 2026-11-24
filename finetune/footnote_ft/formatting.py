"""Turn a dataset row into the exact text the model sees, for training and for evaluation.

Both paths call `render_prompt`, so the fine-tuned model is evaluated on the prompt it was trained
on. The prompt is rendered with the model's own chat template (`enable_thinking=False` for Qwen3,
which ends the prompt with an empty think block); the completion is the JSON verdict, and TRL
appends the tokenizer's end-of-turn token. Training uses TRL's prompt-completion format, so the
loss is computed on the verdict only, not on the passages.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from . import shared


def check_prompt_version(row: Mapping[str, Any]) -> None:
    current = shared.judge_prompt().PROMPT_VERSION
    if row.get("prompt_version") != current:
        raise ValueError(
            f"row {row.get('id')} was built with prompt {row.get('prompt_version')!r}, but the backend's "
            f"judge prompt is now {current!r}: rebuild the dataset (build_dataset.py)"
        )


def system_user(row: Mapping[str, Any]) -> tuple[str, str]:
    """(system, user) for a row, re-rendered from claim + passages with the backend's prompt.

    Re-rendering instead of trusting stored messages means a dataset row and a recorded judgment are
    always shown to the model exactly as the backend would show them today.
    """
    check_prompt_version(row)
    return shared.judge_prompt().render_judge_prompt(str(row["claim"]), row["passages"])


def render_prompt(tokenizer: Any, system: str, user: str, template_kwargs: Mapping[str, Any] | None = None) -> str:
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, **dict(template_kwargs or {})
    )


def target_text(row: Mapping[str, Any]) -> str:
    """The assistant turn: the JSON verdict stored on the row."""
    messages = row["messages"]
    if messages[-1]["role"] != "assistant":
        raise ValueError(f"row {row.get('id')} has no assistant turn")
    return str(messages[-1]["content"])


def to_prompt_completion(
    row: Mapping[str, Any], tokenizer: Any, template_kwargs: Mapping[str, Any] | None = None
) -> dict[str, str]:
    system, user = system_user(row)
    return {"prompt": render_prompt(tokenizer, system, user, template_kwargs), "completion": target_text(row)}
