"""One-request check that a served judge answers in the contract.

    python export/check_served.py --base-url http://localhost:11434/v1 --model footnote-judge

Sends one supported and one unsupported claim (invented text) through the backend's prompt and
prints what came back, whether it parsed, and whether the labels match. For the real numbers run
eval_judge.py with --openai-base-url.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from footnote_ft import shared  # noqa: E402

PASSAGES = [
    {
        "id": "demo:0001",
        "title": "Notes on the Harlow Reading Society",
        "text": "The Harlow Reading Society was founded in 1911 by Ada Linwood. It met every Thursday "
        "evening and lent books to any member who had paid the yearly fee of two dollars.",
    }
]
CASES = [
    ("Ada Linwood founded the Harlow Reading Society in 1911.", "supported"),
    ("The Harlow Reading Society lent books free of charge to anyone in town.", "unsupported"),
]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", default="http://localhost:11434/v1")
    p.add_argument("--model", default="footnote-judge")
    args = p.parse_args(argv)
    jp = shared.judge_prompt()
    ok = True
    for claim, expected in CASES:
        system, user = jp.render_judge_prompt(claim, PASSAGES)
        body = {
            "model": args.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": 0,
            "max_tokens": 256,
        }
        request = urllib.request.Request(
            args.base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        began = time.perf_counter()
        with urllib.request.urlopen(request, timeout=300) as response:
            reply = json.loads(response.read().decode())["choices"][0]["message"]["content"]
        elapsed = (time.perf_counter() - began) * 1000
        parsed = jp.parse_judge_output(reply or "")
        match = parsed is not None and parsed["label"] == expected
        ok &= match
        print(f"[{'ok' if match else 'MISMATCH'}] {elapsed:.0f} ms  expected={expected}  got={parsed}\n  raw={reply!r}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
