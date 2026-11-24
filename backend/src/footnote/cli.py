"""`footnote` command line: ingest the corpus, export data, run evaluations."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from footnote.logs import configure_logging
from footnote.settings import get_settings

SNAPSHOT_SCHOLARS = ["ida-b-wells", "carter-g-woodson", "anna-julia-cooper"]


async def _ingest(args: argparse.Namespace) -> int:
    from footnote.corpus.http import Fetcher
    from footnote.corpus.pipeline import Ingestor, export_chunks, export_snapshot
    from footnote.db.repo import MemoryRepo, MongoRepo, Repo
    from footnote.retrieval.store import PineconeStore, SearchStore

    settings = get_settings()
    repo: Repo
    if args.repo == "memory":
        repo = MemoryRepo()
    else:
        from pymongo import AsyncMongoClient

        repo = MongoRepo(AsyncMongoClient(settings.mongodb_uri.get_secret_value()), settings.mongodb_db)
    store: SearchStore | None = None
    if not args.no_index:
        if settings.pinecone_api_key is None:
            print("PINECONE_API_KEY is not set; use --no-index to only fetch and store.", file=sys.stderr)
            return 2
        store = PineconeStore(
            api_key=settings.pinecone_api_key.get_secret_value(),
            index_name=settings.pinecone_index,
            namespace=settings.pinecone_namespace,
            embed_model=settings.embed_model,
            embed_dimension=settings.embed_dimension,
            rerank_model=settings.rerank_model,
            cloud=settings.pinecone_cloud,
            region=settings.pinecone_region,
        )
    fetcher = Fetcher(
        user_agent=settings.user_agent, cache_dir=settings.data_dir / "raw", has_contact=bool(settings.fetch_contact)
    )
    try:
        ingestor = Ingestor(repo=repo, store=store, fetcher=fetcher, manifest_path=settings.corpus_config)
        report = await ingestor.run(
            only=set(args.only) if args.only else None, papers=not args.no_papers, force=args.force
        )
        print(json.dumps(report.as_dict(), indent=2))
        if args.snapshot:
            counts = await export_snapshot(repo, settings.data_dir / "snapshot.json", SNAPSHOT_SCHOLARS)
            print(f"snapshot: {counts} -> {settings.data_dir / 'snapshot.json'}")
        if args.export_chunks:
            n = await export_chunks(repo, Path(args.export_chunks))
            print(f"exported {n} chunks -> {args.export_chunks}")
    finally:
        await fetcher.close()
        if store is not None:
            await store.close()  # type: ignore[attr-defined]
    return 0


async def _export_chunks(args: argparse.Namespace) -> int:
    from footnote.corpus.pipeline import export_chunks
    from footnote.services import build_services

    services = await build_services(get_settings())
    try:
        n = await export_chunks(services.repo, Path(args.path))
        print(f"exported {n} chunks -> {args.path}")
    finally:
        await services.aclose()
    return 0


async def _export_judgments(args: argparse.Namespace) -> int:
    """Recorded judgments as JSONL for finetune/build_dataset.py --judgments (only human-labelled rows train)."""
    from footnote.services import build_services

    services = await build_services(get_settings())
    n = labelled = 0
    try:
        path = Path(args.path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            async for judgment in services.repo.iter_judgments():
                row = {k: (v.isoformat() if hasattr(v, "isoformat") else v) for k, v in judgment.items()}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                n += 1
                labelled += row.get("human_label") is not None
        print(f"exported {n} judgments ({labelled} human-labelled) -> {path}")
    finally:
        await services.aclose()
    return 0


async def _import_researchers(args: argparse.Namespace) -> int:
    from footnote.corpus.import_researchers import import_researchers

    report = await import_researchers(
        get_settings(),
        source_db=args.source_db,
        apply=args.apply,
        cache_path=Path(args.license_cache),
        report_path=Path(args.report),
        workers=args.workers,
    )
    print(json.dumps(report.as_dict(), indent=2))
    return 0


async def _eval(args: argparse.Namespace) -> int:
    from footnote.evals.run import run_evals

    return await run_evals(args)


def main(argv: list[str] | None = None) -> int:
    configure_logging(stream=sys.stderr)
    parser = argparse.ArgumentParser(prog="footnote")
    sub = parser.add_subparsers(dest="command", required=True)

    ing = sub.add_parser("ingest", help="fetch, check licences, chunk, store and index the corpus")
    ing.add_argument("--only", nargs="*", help="scholar ids to ingest (default: all)")
    ing.add_argument("--no-papers", action="store_true", help="skip the Europe PMC papers")
    ing.add_argument("--no-index", action="store_true", help="store in the database only; do not index in Pinecone")
    ing.add_argument("--force", action="store_true", help="re-chunk and re-index unchanged documents")
    ing.add_argument("--repo", choices=["mongo", "memory"], default="mongo")
    ing.add_argument(
        "--snapshot",
        action="store_true",
        help=f"also write data/snapshot.json for offline mode ({', '.join(SNAPSHOT_SCHOLARS)})",
    )
    ing.add_argument("--export-chunks", metavar="PATH", help="also write every chunk as JSONL (for finetune/)")

    exp = sub.add_parser("export-chunks", help="write every stored chunk as JSONL (input to finetune/build_dataset.py)")
    exp.add_argument("path")

    ej = sub.add_parser("export-judgments", help="write recorded judge verdicts as JSONL (fine-tune data flywheel)")
    ej.add_argument("path")

    mig = sub.add_parser(
        "import-researchers", help="verify licences and import a researcher-profile export into Footnote"
    )
    mig.add_argument(
        "--source-db", required=True, help="MongoDB database holding the export (scholars and passages collections)"
    )
    mig.add_argument("--license-cache", default="data/researcher-license-verification.json")
    mig.add_argument("--report", default="data/researcher-import-report.json")
    mig.add_argument("--workers", type=int, default=16)
    mig.add_argument("--apply", action="store_true", help="write accepted records to MongoDB and Pinecone")

    ev = sub.add_parser("eval", help="run evaluations; writes evals/results/latest.json")
    ev.add_argument("suite", choices=["build", "review", "retrieval", "answers", "judge", "all", "check"])
    ev.add_argument("--dataset", help="override the dataset path")
    ev.add_argument("--limit", type=int, help="only the first N cases (smoke run); for build: how many questions")
    ev.add_argument("--langsmith", action="store_true", help="also upload the run as a LangSmith experiment")
    ev.add_argument("--judges", nargs="*", default=["api"], help="judge backends to compare: api local cascade")

    args = parser.parse_args(argv)
    handler = {
        "ingest": _ingest,
        "export-chunks": _export_chunks,
        "export-judgments": _export_judgments,
        "import-researchers": _import_researchers,
        "eval": _eval,
    }[args.command]
    return asyncio.run(handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
