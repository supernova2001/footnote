"""Model-written claims from an open-weight instruct model, checked before they are kept.

Rules alone cannot paraphrase, so real training runs ask an open-weight model (for example
Qwen/Qwen2.5-7B-Instruct, Apache-2.0) for three variants of each anchor sentence: a faithful
paraphrase, a version that adds one specific detail, and a version that contradicts it. Using an
open-weight model keeps the training data free of closed-model output terms.

The model proposes; the checks below decide. A variant is kept only when the label it is meant to
have can be confirmed against the passage (new numbers or names in a "paraphrase" disqualify it;
an "added detail" that the passage does contain disqualifies it), so labels stay known by
construction even when a model wrote the words.

Two backends: any OpenAI-compatible chat endpoint (vLLM's server, Ollama, llama.cpp, TGI) via the
standard library, or vLLM's offline batch engine (Linux + CUDA, the fastest on a Kaggle GPU).
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Protocol

from .corruptions import PARTIAL, SUPPORTED, UNSUPPORTED, Candidate, _clean_names
from .text import coverage, names, normalize, numbers

GENERATION_VERSION = "gen.v1"

GEN_SYSTEM = """\
You write test claims for training a fact-checking model. You are given a source passage and one \
sentence from it. Write three claims about what that sentence says, as one JSON object:

{"paraphrase": "...", "partial": {"claim": "...", "added": "..."}, "contradiction": {"claim": "...", "changed": "..."}}

- paraphrase: the sentence's meaning in different words. Add nothing: no new names, numbers, dates, \
places, or certainty. Drop nothing that matters.
- partial: the paraphrase plus ONE specific detail the passage does not give (a date, a number, a \
place, a cause, or more certainty, e.g. "suggests" becoming "proves"). "added" is that detail, \
copied exactly as it appears in your claim.
- contradiction: a claim on the same subject that the passage contradicts (a reversal, a wrong \
number, a wrong person). "changed" names what you changed.

Write each claim as one plain declarative sentence, as a writer summarising the source would. \
Reply with the JSON object only."""


def build_generation_prompt(anchor: str, passage_text: str) -> tuple[str, str]:
    user = f"<passage>\n{passage_text.strip()}\n</passage>\n\n<sentence>\n{anchor.strip()}\n</sentence>"
    return GEN_SYSTEM, user


_OBJECT = re.compile(r"\{.*\}", re.DOTALL)
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


def parse_generation(raw: str) -> dict[str, Any] | None:
    match = _OBJECT.search(_THINK.sub("", raw or ""))
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    return data


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _new_numbers(claim: str, passage: str) -> bool:
    present = {m.group(0).replace(",", "") for m in numbers(passage)}
    return any(m.group(0).replace(",", "") not in present for m in numbers(claim))


def _new_names(claim: str, passage: str) -> bool:
    folded = normalize(passage)
    return any(normalize(n) not in folded for n in _clean_names(names(claim)))


def validate_generated(data: dict[str, Any], anchor: str, passage: str) -> list[Candidate]:
    """Keep only the variants whose intended label the passage confirms."""
    kept: list[Candidate] = []
    anchor_norm = normalize(anchor)

    paraphrase = _text(data.get("paraphrase"))
    if (
        paraphrase
        and normalize(paraphrase) != anchor_norm
        and 0.5 <= len(paraphrase) / max(len(anchor), 1) <= 1.8
        and coverage(paraphrase, anchor) >= 0.4
        and not _new_numbers(paraphrase, passage)
        and not _new_names(paraphrase, passage)
    ):
        kept.append(
            Candidate(paraphrase, SUPPORTED, "model_paraphrase", anchor, "The claim restates what the passage says.")
        )

    partial = data.get("partial") if isinstance(data.get("partial"), dict) else {}
    claim, added = _text(partial.get("claim")), _text(partial.get("added"))
    if (
        claim
        and added
        and normalize(added) in normalize(claim)
        and normalize(added) not in normalize(passage)
        and coverage(normalize(claim).replace(normalize(added), " "), anchor) >= 0.5
    ):
        kept.append(
            Candidate(
                claim,
                PARTIAL,
                "model_partial",
                anchor,
                f'The passage supports the core of the claim but not "{added}".',
            )
        )

    contradiction = data.get("contradiction") if isinstance(data.get("contradiction"), dict) else {}
    claim, changed = _text(contradiction.get("claim")), _text(contradiction.get("changed"))
    if (
        claim
        and changed
        and normalize(claim) != anchor_norm
        and normalize(claim) not in normalize(passage)
        and coverage(claim, anchor) >= 0.4
    ):
        kept.append(Candidate(claim, UNSUPPORTED, "model_contradiction", "", "The passage contradicts the claim."))
    return kept


class Generator(Protocol):
    name: str

    def __call__(self, prompts: Sequence[tuple[str, str]]) -> list[str]: ...


class OpenAICompatibleGenerator:
    """POST /chat/completions on any OpenAI-compatible server, standard library only."""

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 600,
        workers: int = 8,
        timeout: float = 180.0,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.name = f"model:{model}"
        self.api_key = api_key or os.environ.get("GENERATOR_API_KEY", "")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.workers = workers
        self.timeout = timeout
        # vLLM's server reads this (e.g. {"enable_thinking": false} for Qwen3); others ignore it.
        self.chat_template_kwargs = chat_template_kwargs

    def _one(self, prompt: tuple[str, str]) -> str:
        system, user = prompt
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.chat_template_kwargs:
            payload["chat_template_kwargs"] = self.chat_template_kwargs
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(self.url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            return str(payload["choices"][0]["message"]["content"] or "")
        except Exception:  # a failed request is a rejected variant, not a crashed build
            return ""

    def __call__(self, prompts: Sequence[tuple[str, str]]) -> list[str]:
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            return list(pool.map(self._one, prompts))


class VLLMGenerator:
    """vLLM's offline engine: one batched call for every prompt. Linux + CUDA (e.g. a Kaggle GPU)."""

    def __init__(
        self,
        model: str,
        *,
        temperature: float = 0.7,
        max_tokens: int = 600,
        chat_template_kwargs: dict[str, Any] | None = None,
        **llm_kwargs: Any,
    ) -> None:
        from vllm import LLM, SamplingParams  # imported here: vLLM is optional and Linux-only

        self.name = f"model:{model}"
        self.llm = LLM(model=model, **llm_kwargs)
        self.params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
        self.chat_template_kwargs = chat_template_kwargs

    def __call__(self, prompts: Sequence[tuple[str, str]]) -> list[str]:
        conversations = [[{"role": "system", "content": s}, {"role": "user", "content": u}] for s, u in prompts]
        kwargs: dict[str, Any] = {"use_tqdm": True}
        if self.chat_template_kwargs:
            kwargs["chat_template_kwargs"] = self.chat_template_kwargs
        outputs = self.llm.chat(conversations, self.params, **kwargs)
        return [o.outputs[0].text if o.outputs else "" for o in outputs]
