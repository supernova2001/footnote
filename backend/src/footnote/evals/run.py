"""`footnote eval {build,review,retrieval,answers,judge,all}`.

Datasets (evals/datasets/):
  ask.candidates.jsonl  questions written by a model from sampled passages (`eval build`)
  ask.jsonl             the reviewed set: only questions a person kept (`eval review`)
Each question: {id, question, kind, gold_passage_ids, scholar_id, expected_abstain, reference}.
  kind: single (one passage answers it), multi (needs two passages), unanswerable (the collection
  does not say), out_of_scope (not about the collection at all)

Suites write rows into evals/results/latest.json (served at GET /api/evals) and a timestamped copy:
  retrieval  recall@5, recall@10, MRR, nDCG@10 per strategy (bm25 / dense / hybrid / hybrid_rerank)
  answers    the full RAG agent per strategy, graded by `evals.grader` (a different model from the
             judge inside the agent): faithfulness, citation precision, relevance, abstention accuracy
  judge      Claude / local / cascade judges on the fine-tune's held-out test split, scored with
             judge_metrics.py, the same function the fine-tune's own eval uses

Every model call is real and billed; `--limit N` runs a smoke version first.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import yaml
from pydantic import BaseModel, Field

from footnote.domain import Passage
from footnote.evals.gate import validate_report
from footnote.evals.metrics import mean, ndcg_at_k, p50, recall_at_k, reciprocal_rank
from footnote.judge_metrics import JudgeCase, eval_row, score_judge
from footnote.llm.client import start_ledger
from footnote.runs import collect, run_ask
from footnote.services import Services, build_services
from footnote.settings import BACKEND_DIR, get_settings

EVALS_DIR = BACKEND_DIR / "evals"
DATASETS = EVALS_DIR / "datasets"
RESULTS = EVALS_DIR / "results"
FINETUNE_DATA = BACKEND_DIR.parent / "finetune" / "data"
STRATEGIES = ("bm25", "dense", "hybrid", "hybrid_rerank")
GATE_POLICY = BACKEND_DIR / "config" / "eval-gates.yaml"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _dataset_provenance(path: Path, rows: list[dict[str, Any]], reviewed_key: str) -> dict[str, Any]:
    row_key = "reviewed" if reviewed_key == "reviewed" else "gold_label"
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "rows": len(rows),
        reviewed_key: bool(rows) and all(bool(row.get(row_key)) for row in rows),
    }


# ── building the question set ────────────────────────────────────────────────────────────────


class WrittenQuestion(BaseModel):
    question: str = Field(description="A question a curious reader would ask; answerable only from the passage(s)")
    reference: str = Field(description="A one- or two-sentence reference answer drawn from the passage(s)")


class UnanswerableQuestions(BaseModel):
    questions: list[str] = Field(description="Plausible questions about this person that these passages do not answer")


QUESTION_SYSTEM = """\
You write evaluation questions for a research assistant. Write one question that the passage(s) \
below answer, as a curious reader would ask it: name the person instead of saying "the author", \
do not quote the passage, and do not ask about page layout. If two passages are given, the \
question must need both. Then give a short reference answer drawn only from the passage(s)."""

UNANSWERABLE_SYSTEM = """\
You write evaluation questions for a research assistant that must say when its sources do not \
cover something. Given a person and a few passages about them, write 3 plausible, specific \
questions about this person that the passages do not answer (for example about things after the \
period they cover, or private details no source would record). They must sound like real questions."""


async def build_questions(services: Services, n: int, seed: int = 7) -> Path:
    rng = random.Random(seed)
    chunks = [c async for c in services.repo.iter_chunks() if len(c.text) > 400]
    by_scholar: dict[str, list[Any]] = {}
    for c in chunks:
        by_scholar.setdefault(c.scholar_ids[0], []).append(c)
    rows: list[dict[str, Any]] = []
    n_single, n_multi, n_unans = int(n * 0.55), int(n * 0.25), max(1, int(n * 0.2) // 3)
    for i, c in enumerate(rng.sample(chunks, min(n_single, len(chunks)))):
        q = await services.models.structured(
            "evals.questions",
            system=QUESTION_SYSTEM,
            output=WrittenQuestion,
            messages=[{"role": "user", "content": f"[{c.id}] {c.title}\n{c.text}"}],
        )
        rows.append(
            {
                "id": f"single-{i:03d}",
                "question": q.question,
                "kind": "single",
                "gold_passage_ids": [c.id],
                "scholar_id": None,
                "expected_abstain": False,
                "reference": q.reference,
            }
        )
    pools = [v for v in by_scholar.values() if len(v) >= 2]
    for i in range(min(n_multi, len(pools) * 4)):
        a, b = rng.sample(rng.choice(pools), 2)
        q = await services.models.structured(
            "evals.questions",
            system=QUESTION_SYSTEM,
            output=WrittenQuestion,
            messages=[{"role": "user", "content": f"[{a.id}] {a.text}\n\n[{b.id}] {b.text}"}],
        )
        rows.append(
            {
                "id": f"multi-{i:03d}",
                "question": q.question,
                "kind": "multi",
                "gold_passage_ids": [a.id, b.id],
                "scholar_id": None,
                "expected_abstain": False,
                "reference": q.reference,
            }
        )
    scholars = [s for s in await services.repo.list_scholars() if s.kind == "historical"]
    for i, s in enumerate(rng.sample(scholars, min(n_unans, len(scholars)))):
        sample = rng.sample(by_scholar.get(s.id, []), min(4, len(by_scholar.get(s.id, []))))
        out = await services.models.structured(
            "evals.questions",
            system=UNANSWERABLE_SYSTEM,
            output=UnanswerableQuestions,
            messages=[{"role": "user", "content": f"Person: {s.name}\n\n" + "\n\n".join(c.text for c in sample)}],
        )
        for j, question in enumerate(out.questions[:3]):
            rows.append(
                {
                    "id": f"unanswerable-{i:02d}-{j}",
                    "question": question,
                    "kind": "unanswerable",
                    "gold_passage_ids": [],
                    "scholar_id": None,
                    "expected_abstain": True,
                    "reference": "",
                }
            )
    path = DATASETS / "ask.candidates.jsonl"
    _write_jsonl(path, rows)
    return path


def review_questions() -> Path:
    """Keep or drop each candidate by hand: an eval set nobody read is not an eval set."""
    src, dst = DATASETS / "ask.candidates.jsonl", DATASETS / "ask.jsonl"
    kept = _read_jsonl(dst) if dst.exists() else []
    seen = {r["id"] for r in kept}
    for row in _read_jsonl(src):
        if row["id"] in seen:
            continue
        print(
            f"\n[{row['kind']}] {row['question']}\n  reference: {row['reference']}\n  gold: {row['gold_passage_ids']}"
        )
        choice = input("  keep / drop / edit / quit [k/d/e/q]: ").strip().lower()
        if choice == "q":
            break
        if choice == "e":
            row["question"] = input("  question: ").strip() or row["question"]
            choice = "k"
        if choice == "k":
            kept.append({**row, "reviewed": True})
    _write_jsonl(dst, kept)
    return dst


# ── suites ───────────────────────────────────────────────────────────────────────────────────


async def eval_retrieval(services: Services, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cases = [r for r in rows if r["gold_passage_ids"]]
    out = []
    for strategy in STRATEGIES:
        r5, r10, rr, nd = [], [], [], []
        for row in cases:
            ranked = [p.id for p in await services.retriever.search(row["question"], strategy=strategy, k=10)]  # type: ignore[arg-type]
            r5.append(recall_at_k(ranked, row["gold_passage_ids"], 5))
            r10.append(recall_at_k(ranked, row["gold_passage_ids"], 10))
            rr.append(reciprocal_rank(ranked, row["gold_passage_ids"]))
            nd.append(ndcg_at_k(ranked, row["gold_passage_ids"], 10))
        out.append(
            {
                "strategy": strategy,
                "n": len(cases),
                "recall_at_5": mean(r5),
                "recall_at_10": mean(r10),
                "mrr": mean(rr),
                "ndcg_at_10": mean(nd),
            }
        )
        print(f"retrieval {strategy:14} " + " ".join(f"{k}={v}" for k, v in out[-1].items() if k != "strategy"))
    return out


class AnswerGrade(BaseModel):
    faithfulness: float = Field(
        ge=0, le=1, description="Share of the answer's factual sentences supported by the passages"
    )
    citation_precision: float = Field(
        ge=0, le=1, description="Share of citations whose passage supports the sentence citing it"
    )
    relevance: int = Field(ge=1, le=5, description="How directly the answer addresses the question")
    note: str


GRADER_SYSTEM = """\
You grade a research assistant's answer. You see the question, the passages it retrieved (with \
ids), and the answer, whose sentences end with the ids they cite. Grade only against the passages.
- faithfulness: of the answer's factual sentences, the share fully supported by the passages.
- citation_precision: of the citations, the share whose passage actually supports the sentence.
- relevance: 1 (off the point) to 5 (answers exactly what was asked). An answer that correctly \
says the sources do not cover the question is relevant (5) when they indeed do not."""


async def eval_answers(
    services: Services, rows: list[dict[str, Any]], strategies: tuple[str, ...]
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    summary, per_case = [], {}
    for strategy in strategies:
        results = []
        for row in rows:
            ledger, started = start_ledger(), time.perf_counter()
            out = await collect(
                run_ask(
                    services, question=row["question"], scholar_id=row.get("scholar_id"), strategy=cast(Any, strategy)
                )
            )
            latency = (time.perf_counter() - started) * 1000
            cost = ledger.usage.cost_usd
            answer = out.get("answer", {"segments": [], "abstained": True})
            passages = out.get("passages", {}).get("passages", [])
            rendered = " ".join(
                s["text"].strip() + "".join(f" [{c['passage_id']}]" for c in s["citations"]) for s in answer["segments"]
            )
            context = "\n\n".join(f"[{p['id']}] {p['text']}" for p in passages) or "(no passages)"
            grade = await services.models.structured(
                "evals.grader",
                system=GRADER_SYSTEM,
                output=AnswerGrade,
                messages=[
                    {
                        "role": "user",
                        "content": f"Question: {row['question']}\n\nPassages:\n{context}\n\nAnswer:\n{rendered}",
                    }
                ],
            )
            results.append(
                {
                    "id": row["id"],
                    "question": row["question"],
                    "kind": row["kind"],
                    "answer": rendered,
                    "abstained": answer["abstained"],
                    "expected_abstain": row["expected_abstain"],
                    "grade": grade.model_dump(),
                    "latency_ms": latency,
                    "cost_usd": cost,
                }
            )
        per_case[strategy] = results
        summary.append(
            {
                "strategy": strategy,
                "config": f"{strategy} + judge({services.judges.name})",
                "n": len(results),
                "faithfulness": mean([r["grade"]["faithfulness"] for r in results if not r["abstained"]]),
                "citation_precision": mean([r["grade"]["citation_precision"] for r in results if not r["abstained"]]),
                "answer_relevance": mean([(r["grade"]["relevance"] - 1) / 4 for r in results]),
                "abstention_accuracy": mean([float(r["abstained"] == r["expected_abstain"]) for r in results]),
                "p50_latency_ms": round(p50([r["latency_ms"] for r in results]), 1),
                "cost_per_answer_usd": mean([r["cost_usd"] for r in results]),
            }
        )
        print("answers   " + " ".join(f"{k}={v}" for k, v in summary[-1].items()))
    return summary, per_case


def _passages(row: dict[str, Any]) -> list[Passage]:
    return [
        Passage(
            id=p["id"],
            document_id=row.get("document_id", ""),
            title=p.get("title", ""),
            section=None,
            text=p["text"],
            scholar_ids=[],
            license_id="PD-US",
            source_url="",
        )
        for p in row["passages"]
    ]


async def eval_judges(services: Services, rows: list[dict[str, Any]], backends: list[str]) -> list[dict[str, Any]]:
    """Score each judge backend on the same rows. Replies are raw (before the agents' fail-closed rule);
    judge_metrics reports both the raw and the as-deployed scores."""
    from footnote.agents.judge import ApiJudge, CascadeJudge, LocalJudge

    settings = services.settings
    api = ApiJudge(services.models)
    local = LocalJudge(settings.local_judge_url, settings.local_judge_model) if settings.local_judge_url else None
    sem = asyncio.Semaphore(services.config.judge.concurrency)
    out = []
    for backend in backends:
        if backend != "api" and local is None:
            print(f"judge {backend}: skipped (set LOCAL_JUDGE_URL to the served fine-tune; see finetune/export)")
            continue
        cascade = CascadeJudge(local, api, services.config.judge.escalate_on) if local else None
        ask: Any
        if backend == "api":
            ask, name = api.raw_reply, api.name
        elif backend == "local":
            assert local is not None
            ask, name = local.raw_reply, local.name
        else:
            assert cascade is not None
            name = cascade.name

            async def ask(claim: str, passages: list[Passage], cascade: CascadeJudge = cascade) -> str:
                ruling = await cascade.rule(claim, passages)
                return json.dumps(
                    {
                        **ruling.verdict.model_dump(include={"label", "evidence", "rationale"}),
                        "escalated": ruling.escalated,
                    }
                )

        ledger = start_ledger()

        async def one(row: dict[str, Any], ask: Any = ask) -> JudgeCase:
            async with sem:
                started = time.perf_counter()
                reply = await ask(row["claim"], _passages(row))
                latency = (time.perf_counter() - started) * 1000
            return JudgeCase(
                gold=row["label"], reply=reply, passages=row["passages"], latency_ms=latency, id=str(row["id"])
            )

        cases = list(await asyncio.gather(*(one(r) for r in rows)))
        result = eval_row(score_judge(cases), name, cost_per_1k_usd=ledger.usage.cost_usd / len(cases) * 1000)
        result["backend"] = backend
        if backend == "cascade":
            result["escalation_rate"] = round(sum('"escalated": true' in c.reply for c in cases) / len(cases), 4)
        out.append(result)
        print("judge     " + " ".join(f"{k}={v}" for k, v in result.items()))
    return out


# ── LangSmith experiments (optional) ────────────────────────────────────────────────────────


async def upload_to_langsmith(per_case: dict[str, list[dict[str, Any]]]) -> None:
    """Record finished answer runs as LangSmith experiments, without re-running the agent."""
    from langsmith import Client

    client = Client()
    name = "footnote-ask"
    first = next(iter(per_case.values()))
    if not client.has_dataset(dataset_name=name):
        ds = client.create_dataset(name, description="Footnote RAG eval questions (reviewed)")
        client.create_examples(
            dataset_id=ds.id,
            inputs=[{"id": r["id"], "question": r["question"]} for r in first],
            outputs=[{"expected_abstain": r["expected_abstain"]} for r in first],
        )
    for strategy, results in per_case.items():
        by_id = {r["id"]: r for r in results}

        async def target(inputs: dict[str, Any], by_id: dict[str, Any] = by_id) -> dict[str, Any]:
            return by_id[inputs["id"]]

        def scores(outputs: dict[str, Any], reference_outputs: dict[str, Any]) -> list[dict[str, Any]]:
            g = outputs["grade"]
            return [
                {"key": "faithfulness", "score": g["faithfulness"]},
                {"key": "citation_precision", "score": g["citation_precision"]},
                {"key": "relevance", "score": (g["relevance"] - 1) / 4},
                {
                    "key": "abstention_correct",
                    "score": float(outputs["abstained"] == reference_outputs["expected_abstain"]),
                },
            ]

        await client.aevaluate(
            target,
            data=name,
            evaluators=[scores],  # type: ignore[list-item]
            experiment_prefix=f"ask-{strategy}",  # type: ignore[arg-type]
            metadata={"strategy": strategy},
        )


# ── entry point ──────────────────────────────────────────────────────────────────────────────


async def run_evals(args: argparse.Namespace) -> int:
    settings = get_settings()
    if args.suite == "check":
        latest = RESULTS / "latest.json"
        if not latest.exists():
            print(f"evaluation gate failed: {latest} does not exist")
            return 1
        policy = yaml.safe_load(GATE_POLICY.read_text(encoding="utf-8"))
        errors = validate_report(json.loads(latest.read_text(encoding="utf-8")), policy)
        if errors:
            print("evaluation gate failed:\n- " + "\n- ".join(errors))
            return 1
        print("evaluation gate passed")
        return 0
    services = await build_services(settings)
    try:
        if args.suite == "build":
            print(f"wrote {await build_questions(services, args.limit or 120)}; next: footnote eval review")
            return 0
        if args.suite == "review":
            print(f"kept -> {review_questions()}")
            return 0
        # Fakes exercise the pipeline but measure nothing: their numbers never reach /evals.
        if services.offline:
            latest = RESULTS / "offline-latest.json"
        elif args.suite == "all" and args.limit is None:
            latest = RESULTS / "latest.json"
        else:
            suffix = "-smoke" if args.limit is not None else ""
            latest = RESULTS / f"{args.suite}{suffix}-latest.json"
        if services.offline:
            print("offline mode: fake models; results go to offline-latest.json and are not quality numbers")
        report: dict[str, Any] = (
            json.loads(latest.read_text()) if latest.exists() else {"retrieval": [], "answers": [], "judge": []}
        )
        answer_path: Path | None = None
        answer_rows: list[dict[str, Any]] = []
        judge_path_used: Path | None = None
        judge_set: list[dict[str, Any]] = []
        if args.suite in ("retrieval", "answers", "all"):
            path = Path(args.dataset) if args.dataset else DATASETS / "ask.jsonl"
            rows = _read_jsonl(path)[: args.limit] if args.limit else _read_jsonl(path)
            answer_path, answer_rows = path, rows
            if args.suite in ("retrieval", "all"):
                report["retrieval"] = await eval_retrieval(services, rows)
            if args.suite in ("answers", "all"):
                report["answers"], per_case = await eval_answers(services, rows, ("bm25", "hybrid_rerank"))
                _write_jsonl(RESULTS / f"answers-{_stamp()}.jsonl", [r for rs in per_case.values() for r in rs])
                if args.langsmith:
                    await upload_to_langsmith(per_case)
        if args.suite in ("judge", "all"):
            judge_path = Path(args.dataset) if args.dataset and args.suite == "judge" else None
            judge_set = judge_rows(judge_path)
            judge_path_used = judge_path or (
                FINETUNE_DATA / "test_gold.jsonl"
                if any(row.get("gold_label") for row in judge_set)
                else FINETUNE_DATA / "test.jsonl"
            )
            report["judge"] = await eval_judges(
                services, judge_set[: args.limit] if args.limit else judge_set, args.judges
            )
        report["generated_at"] = datetime.now(UTC).isoformat()
        report["mode"] = "offline" if services.offline else "production"
        report["run_scope"] = args.suite
        report["provenance"] = {
            "corpus": await services.repo.counts(),
            "models": {name: services.models.model_for(name) for name in services.config.routes},
            "answer_dataset": _dataset_provenance(answer_path, answer_rows, "reviewed") if answer_path else {},
            "judge_dataset": _dataset_provenance(judge_path_used, judge_set, "human_labeled")
            if judge_path_used
            else {},
        }
        RESULTS.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(report, indent=2)
        latest.write_text(payload)
        (RESULTS / f"report-{_stamp()}.json").write_text(payload)
        print(f"wrote {latest}")
        return 0
    finally:
        await services.aclose()


def judge_rows(path: Path | None = None) -> list[dict[str, Any]]:
    """The judge test set: the human-labelled gold rows when any are labelled, else the held-out split.

    finetune/ writes `test.jsonl` (label = known by construction) and `test_gold.jsonl` (gold_label =
    set by a person with label_gold.py, null until then). Gold rows are preferred: a label known by
    construction can still be wrong, a reviewed one is the closest thing to ground truth.
    """
    if path is not None:
        return [{**r, "label": r.get("gold_label") or r["label"]} for r in _read_jsonl(path)]
    gold = FINETUNE_DATA / "test_gold.jsonl"
    if gold.exists():
        labelled = [{**r, "label": r["gold_label"]} for r in _read_jsonl(gold) if r.get("gold_label")]
        if labelled:
            print(f"judge test set: {gold} ({len(labelled)} human-labelled rows)")
            return labelled
    test = FINETUNE_DATA / "test.jsonl"
    if test.exists():
        print(f"judge test set: {test} (labels by construction; label test_gold.jsonl for a reviewed set)")
        return _read_jsonl(test)
    raise FileNotFoundError(f"No judge test set in {FINETUNE_DATA}; run finetune/build_dataset.py first")


def _stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:4]
