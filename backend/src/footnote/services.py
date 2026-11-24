"""Builds and wires every component once per process. Storage, providers and offline differ only here."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import structlog
from anthropic import AsyncAnthropic

from footnote.agents.ask import AskDeps, build_ask_graph
from footnote.agents.judge import ApiJudge, CascadeJudge, Judge, JudgePool, LocalJudge, RecordingJudge
from footnote.agents.mcp_tools import load_external_servers
from footnote.agents.tool_agents import OfflineResearchAgent, OfflineSupervisor, ResearchAgent, Supervisor
from footnote.agents.writer import WriterDeps, build_writer_graph
from footnote.db.repo import MemoryRepo, MongoRepo, Repo
from footnote.llm.client import ClaudeModels, Models
from footnote.llm.config import ModelsConfig, load_models_config
from footnote.llm.fake import FakeModels
from footnote.llm.router import RoutedModels
from footnote.retrieval.retriever import Retriever
from footnote.retrieval.store import MemoryStore, PineconeStore, SearchStore
from footnote.settings import Settings

log = structlog.get_logger(__name__)


@dataclass
class Services:
    settings: Settings
    config: ModelsConfig
    offline: bool
    repo: Repo
    store: SearchStore
    models: Models
    retriever: Retriever
    judges: JudgePool
    checkpointer: Any
    anthropic: AsyncAnthropic | None = None
    ask_graph: Any = None
    writer_graph: Any = None
    researcher: Any = None
    supervisor: Any = None
    closers: list[Callable[[], Awaitable[None]]] = field(default_factory=list)

    def mcp_server(self) -> Any:
        from footnote.mcp_server.server import build_mcp_server

        return build_mcp_server(self)

    async def aclose(self) -> None:
        for close in reversed(self.closers):
            try:
                await close()
            except Exception:
                log.warning("close_failed", exc_info=True)


def build_judges(
    config: ModelsConfig, models: Models, settings: Settings, repo: Repo, backend: str | None = None
) -> JudgePool:
    backend = backend or settings.judge_backend or config.judge.backend
    api: Judge = ApiJudge(models)
    inner: Judge
    if backend == "api":
        inner = api
    else:
        if not settings.local_judge_url:
            raise RuntimeError(f"judge backend {backend!r} needs LOCAL_JUDGE_URL (see finetune/export)")
        local = LocalJudge(settings.local_judge_url, settings.local_judge_model)
        inner = local if backend == "local" else CascadeJudge(local, api, config.judge.escalate_on)
    return JudgePool(RecordingJudge(inner, repo), config.judge.concurrency)


async def build_services(settings: Settings) -> Services:
    """Wire storage and models independently:

    FOOTNOTE_OFFLINE=1        fake models + in-memory storage (tests, CI, keyless demo)
    FOOTNOTE_STORAGE=local    real models + in-memory storage over the bundled snapshot
    default                   real models + MongoDB + Pinecone
    """
    config = load_models_config(settings.models_config, settings.llm_provider)
    offline = settings.offline
    closers: list[Callable[[], Awaitable[None]]] = []
    repo, store, checkpointer = await _storage(settings, closers, local=offline or settings.storage == "local")

    anthropic_client: Any = None
    openai_client: Any = None
    models: Models
    if offline:
        models = FakeModels(config)
    else:
        providers: dict[str, Models] = {}
        if "anthropic" in config.providers():
            anthropic_client = AsyncAnthropic(max_retries=3)
            closers.append(anthropic_client.close)
            providers["anthropic"] = ClaudeModels(config, anthropic_client)
        if "openai" in config.providers():
            from openai import AsyncOpenAI

            from footnote.llm.openai_client import OpenAIModels

            if settings.openai_api_key is None:
                raise RuntimeError("models.yaml routes use OpenAI but OPENAI_API_KEY is not set")
            openai_client = AsyncOpenAI(api_key=settings.openai_api_key.get_secret_value(), max_retries=3)
            closers.append(openai_client.close)
            providers["openai"] = OpenAIModels(config, openai_client)
        models = RoutedModels(config, providers)

    retriever = Retriever(store, repo)
    judges = build_judges(config, models, settings, repo)
    services = Services(
        settings=settings,
        config=config,
        offline=offline,
        repo=repo,
        store=store,
        models=models,
        retriever=retriever,
        judges=judges,
        checkpointer=checkpointer,
        anthropic=anthropic_client,
        closers=closers,
    )
    services.ask_graph = build_ask_graph(AskDeps(models=models, retriever=retriever, judges=judges, repo=repo))
    external = load_external_servers(settings.mcp_servers_config)
    research_provider = config.route("research.agent").provider
    chat_provider = config.route("chat.supervisor").provider
    if offline:
        services.researcher = OfflineResearchAgent(repo, services.mcp_server)
        services.supervisor = OfflineSupervisor(services.mcp_server)
    else:
        from footnote.agents.openai_agents import OpenAIResearchAgent, OpenAISupervisor

        services.researcher = (
            OpenAIResearchAgent(openai_client, config, repo, services.mcp_server)
            if research_provider == "openai"
            else ResearchAgent(anthropic_client, config, repo, services.mcp_server)
        )
        services.supervisor = (
            OpenAISupervisor(openai_client, config, services.mcp_server, external)
            if chat_provider == "openai"
            else Supervisor(anthropic_client, config, services.mcp_server, external)
        )
    services.writer_graph = build_writer_graph(
        WriterDeps(
            models=models, judges=judges, retriever=retriever, repo=repo, find_sources=services.researcher.find_sources
        ),
        checkpointer,
    )
    return services


async def _storage(
    settings: Settings, closers: list[Callable[[], Awaitable[None]]], *, local: bool
) -> tuple[Repo, SearchStore, Any]:
    if local:
        from langgraph.checkpoint.memory import InMemorySaver

        from footnote.corpus.pipeline import load_snapshot

        repo, store = MemoryRepo(), MemoryStore()
        snapshot = settings.data_dir / "snapshot.json"
        if snapshot.exists():
            log.info("local_snapshot_loaded", **await load_snapshot(repo, store, snapshot))
        else:
            log.warning("local_without_snapshot", hint="run `footnote ingest --repo memory --snapshot` once")
        return repo, store, InMemorySaver()

    from langgraph.checkpoint.mongodb import MongoDBSaver
    from pymongo import AsyncMongoClient, MongoClient

    uri = settings.mongodb_uri.get_secret_value()
    mongo: AsyncMongoClient[dict[str, Any]] = AsyncMongoClient(uri, appname="footnote")
    closers.append(mongo.close)
    sync_mongo: MongoClient[Any] = MongoClient(uri, appname="footnote-checkpoints")

    async def close_sync() -> None:
        sync_mongo.close()

    closers.append(close_sync)
    if settings.pinecone_api_key is None:
        raise RuntimeError("PINECONE_API_KEY is not set (or use FOOTNOTE_STORAGE=local / FOOTNOTE_OFFLINE=1)")
    pinecone_store = PineconeStore(
        api_key=settings.pinecone_api_key.get_secret_value(),
        index_name=settings.pinecone_index,
        namespace=settings.pinecone_namespace,
        embed_model=settings.embed_model,
        embed_dimension=settings.embed_dimension,
        rerank_model=settings.rerank_model,
        cloud=settings.pinecone_cloud,
        region=settings.pinecone_region,
    )
    closers.append(pinecone_store.close)
    checkpointer = MongoDBSaver(sync_mongo, db_name=settings.mongodb_db, checkpoint_collection_name="checkpoints")
    return MongoRepo(mongo, settings.mongodb_db), pinecone_store, checkpointer
