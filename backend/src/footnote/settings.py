"""Configuration from the environment (and `.env`). Secrets are never logged or returned by the API."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[2]
CONFIG_DIR = BACKEND_DIR / "config"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(BACKEND_DIR.parent / ".env", BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Offline mode serves deterministic fakes for the model, the vector store and the database,
    # so the UI, the tests and CI run with no API keys and no network.
    offline: bool = Field(default=False, alias="FOOTNOTE_OFFLINE")
    # Where documents, vectors and graph checkpoints live when the models are real:
    #   cloud - MongoDB + Pinecone (production)
    #   local - in memory, loaded from the bundled public-domain snapshot (try real models with no database)
    storage: Literal["cloud", "local"] = Field(default="cloud", alias="FOOTNOTE_STORAGE")

    # Model providers. Each route in models.yaml names its provider; FOOTNOTE_LLM_PROVIDER switches
    # every route to a preset from models.yaml `presets`: anthropic (the routes as written, Claude),
    # openai (cheapest OpenAI tiers) or openai-quality (OpenAI mid/top tiers).
    anthropic_api_key: SecretStr | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    openai_api_key: SecretStr | None = Field(default=None, alias="OPENAI_API_KEY")
    llm_provider: str | None = Field(default=None, alias="FOOTNOTE_LLM_PROVIDER")
    models_config: Path = Field(default=CONFIG_DIR / "models.yaml", alias="FOOTNOTE_MODELS_CONFIG")

    # MongoDB: documents, chunks, drafts, judgments, conversations, LangGraph checkpoints.
    mongodb_uri: SecretStr = Field(default=SecretStr("mongodb://localhost:27017"), alias="MONGODB_URI")
    mongodb_db: str = Field(default="footnote", alias="MONGODB_DB")

    # Pinecone: one document index with a BM25 full-text field and an integrated-embedding field.
    pinecone_api_key: SecretStr | None = Field(default=None, alias="PINECONE_API_KEY")
    pinecone_index: str = Field(default="footnote", alias="PINECONE_INDEX")
    pinecone_namespace: str = Field(default="corpus", alias="PINECONE_NAMESPACE")
    pinecone_cloud: str = Field(default="aws", alias="PINECONE_CLOUD")
    pinecone_region: str = Field(default="us-east-1", alias="PINECONE_REGION")
    embed_model: str = Field(default="llama-text-embed-v2", alias="PINECONE_EMBED_MODEL")
    embed_dimension: int = Field(default=1024, alias="PINECONE_EMBED_DIMENSION")
    rerank_model: str = Field(default="bge-reranker-v2-m3", alias="PINECONE_RERANK_MODEL")

    # The fine-tuned judge, served by Ollama or llama.cpp behind an OpenAI-compatible API.
    local_judge_url: str | None = Field(default=None, alias="LOCAL_JUDGE_URL")
    local_judge_model: str = Field(default="footnote-judge", alias="LOCAL_JUDGE_MODEL")
    # Overrides `judge.backend` in models.yaml: claude | local | cascade.
    judge_backend: str | None = Field(default=None, alias="FOOTNOTE_JUDGE_BACKEND")

    # Corpus fetching. The contact goes into the User-Agent only if the operator sets it.
    corpus_config: Path = Field(default=CONFIG_DIR / "corpus.yaml", alias="FOOTNOTE_CORPUS_CONFIG")
    data_dir: Path = Field(default=BACKEND_DIR / "data", alias="FOOTNOTE_DATA_DIR")
    fetch_contact: str | None = Field(default=None, alias="FOOTNOTE_FETCH_CONTACT")

    # MCP servers the research agent may attach besides Footnote's own (JSON, see config/).
    mcp_servers_config: Path = Field(default=CONFIG_DIR / "mcp_servers.json", alias="FOOTNOTE_MCP_SERVERS")

    # Public-demo protection: per-client rate limit and a daily spend ceiling on model calls.
    cors_origins: list[str] = Field(default=["http://localhost:3000"], alias="CORS_ORIGINS")
    # Host headers the MCP endpoint accepts when deployed (DNS-rebinding protection); empty = localhost only.
    mcp_allowed_hosts: list[str] = Field(default=[], alias="MCP_ALLOWED_HOSTS")
    rate_limit_per_minute: int = Field(default=12, alias="RATE_LIMIT_PER_MINUTE")
    daily_budget_usd: float = Field(default=5.0, alias="DAILY_BUDGET_USD")

    # JSON object mapping stable owner ids to bearer API keys. When configured, every route that
    # spends tokens and every draft route requires a matching key. Empty keeps local/demo mode open.
    api_keys_json: SecretStr | None = Field(default=None, alias="FOOTNOTE_API_KEYS")

    langsmith_project: str = Field(default="footnote", alias="LANGSMITH_PROJECT")

    @property
    def user_agent(self) -> str:
        contact = f"; contact: {self.fetch_contact}" if self.fetch_contact else ""
        return f"Footnote/0.1 (educational research demo{contact})"

    @property
    def api_keys(self) -> dict[str, str]:
        if self.api_keys_json is None:
            return {}
        value = json.loads(self.api_keys_json.get_secret_value())
        if not isinstance(value, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in value.items()):
            raise ValueError("FOOTNOTE_API_KEYS must be a JSON object mapping owner ids to API keys")
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()
