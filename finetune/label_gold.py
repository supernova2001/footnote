"""Hand-label the gold test sample, one claim at a time, in the terminal.

    python label_gold.py data/out/test_gold.jsonl

Keys: s = supported, p = partial, u = unsupported, k = skip, n = add a note, q = quit.
The constructed label stays hidden while you label (so it cannot anchor you); each answer is saved
immediately, and at the end you see how often the construction agreed with you, which is the
measured quality of the dataset's labels.
"""

from __future__ import annotations

import argparse
import textwrap
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from footnote_ft.io import read_jsonl, write_jsonl

KEYS = {"s": "supported", "p": "partial", "u": "unsupported"}


def apply_label(rows: list[dict[str, Any]], index: int, key: str) -> bool:
    """Record a label for rows[index]; True if the key was a label."""
    label = KEYS.get(key.strip().lower())
    if label is None:
        return False
    rows[index]["gold_label"] = label
    return True


def agreement(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    labelled = [r for r in rows if r.get("gold_label")]
    if not labelled:
        return {"labelled": 0, "agreement": None, "disagreements": {}}
    agree = sum(1 for r in labelled if r["gold_label"] == r["constructed_label"])
    disagreements = Counter(
        f"{r['constructed_label']}->{r['gold_label']} ({r['method']})"
        for r in labelled
        if r["gold_label"] != r["constructed_label"]
    )
    return {
        "labelled": len(labelled),
        "agreement": agree / len(labelled),
        "disagreements": dict(disagreements.most_common()),
    }


def show(row: Mapping[str, Any], position: int, total: int) -> None:
    print("\n" + "=" * 78)
    print(f"[{position}/{total}]  {row['id']}")
    for passage in row["passages"]:
        print(f"\n--- passage {passage['id']}  ({passage.get('title', '')})")
        print(textwrap.fill(passage["text"], width=78))
    print("\n>>> CLAIM")
    print(textwrap.fill(row["claim"], width=78))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", type=Path)
    args = parser.parse_args(argv)
    rows = list(read_jsonl(args.path))
    todo = [i for i, r in enumerate(rows) if not r.get("gold_label")]
    print(f"{len(rows) - len(todo)} of {len(rows)} already labelled.")
    for n, index in enumerate(todo, 1):
        show(rows[index], n, len(todo))
        while True:
            key = input("\n[s]upported [p]artial [u]nsupported  [k] skip  [n] note  [q] quit > ").strip().lower()
            if key == "q":
                write_jsonl(args.path, rows)
                print(agreement(rows))
                return 0
            if key == "k":
                break
            if key == "n":
                rows[index]["note"] = input("note: ").strip()
                write_jsonl(args.path, rows)
                continue
            if apply_label(rows, index, key):
                write_jsonl(args.path, rows)
                break
            print("use s, p, u, k, n or q")
    print(agreement(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
