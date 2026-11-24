"""HTTP API: REST for the collection, server-sent events for the agents, MCP at /mcp."""

from __future__ import annotations

import json
import secrets
import time
from collections import defaultdict, deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal

import structlog
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse
from starlette.responses import JSONResponse

from footnote.auth import current_owner
from footnote.domain import Audience, Passage, Strategy
from footnote.llm.client import ModelRefusal
from footnote.logs import configure_logging
from footnote.runs import Event, run_ask, run_chat, run_draft_review, run_draft_start
from footnote.services import Services, build_services
from footnote.settings import BACKEND_DIR, Settings, get_settings

log = structlog.get_logger(__name__)
EVALS_LATEST = BACKEND_DIR / "evals" / "results" / "latest.json"


def _authenticate(keys: dict[str, str], authorization: str) -> str:
    if not keys:
        return "anonymous"
    scheme, _, supplied = authorization.partition(" ")
    if scheme.lower() != "bearer" or not supplied:
        raise HTTPException(401, "Bearer API key required", headers={"WWW-Authenticate": "Bearer"})
    owner = next((owner for owner, key in keys.items() if secrets.compare_digest(supplied, key)), None)
    if owner is None:
        raise HTTPException(401, "Invalid API key", headers={"WWW-Authenticate": "Bearer"})
    return owner


class MCPAuthMiddleware:
    """Protect the mounted MCP ASGI app, which cannot use FastAPI dependencies."""

    def __init__(self, app: Any, keys: dict[str, str]) -> None:
        self.app = app
        self.keys = keys

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http" or not str(scope.get("path", "")).startswith("/mcp"):
            await self.app(scope, receive, send)
            return
        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers", [])}
        try:
            owner_id = _authenticate(self.keys, headers.get("authorization", ""))
        except HTTPException as exc:
            await JSONResponse({"detail": exc.detail}, status_code=exc.status_code, headers=exc.headers)(
                scope, receive, send
            )
            return
        token = current_owner.set(owner_id)
        try:
            await self.app(scope, receive, send)
        finally:
            current_owner.reset(token)


# ── request bodies ──────────────────────────────────────────────────────────────────────────


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    scholar_id: str | None = None
    strategy: Strategy = "hybrid_rerank"
    # Earlier turns of the chat, oldest first, so follow-ups ("what about Woodson?") can be resolved.
    history: list[ChatMessage] = Field(default=[], max_length=12)


class DraftRequest(BaseModel):
    brief: str = Field(min_length=10, max_length=1000)
    audience: Audience = "grade8"
    scholar_id: str | None = None
    document_id: str | None = None


class ReviewRequest(BaseModel):
    decision: Literal["approve", "revise"]
    notes: str | None = Field(default=None, max_length=2000)


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1, max_length=20)


# ── protection for a public demo: per-client rate limit + daily spend ceiling ──────────────


class RateLimiter:
    """Sliding one-minute window per client. In-process: one instance per replica (documented)."""

    def __init__(self, per_minute: int) -> None:
        self.per_minute = per_minute
        self.hits: dict[str, deque[float]] = defaultdict(deque)

    def check(self, key: str) -> None:
        now = time.monotonic()
        window = self.hits[key]
        while window and now - window[0] > 60:
            window.popleft()
        if len(window) >= self.per_minute:
            raise HTTPException(429, f"Rate limit: {self.per_minute} agent runs per minute. Try again shortly.")
        window.append(now)


def _client_key(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "")
    return forwarded.split(",")[0].strip() or (request.client.host if request.client else "unknown")


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    raise TypeError(f"not JSON serialisable: {type(value).__name__}")


def _sse(events: AsyncIterator[Event]) -> EventSourceResponse:
    async def gen() -> AsyncIterator[dict[str, str]]:
        try:
            async for event, data in events:
                if not event.startswith("__"):
                    yield {"event": event, "data": json.dumps(data, default=_json_default)}
        except ModelRefusal as exc:
            yield {
                "event": "error",
                "data": json.dumps({"message": "The model declined this request.", "category": exc.category}),
            }
        except Exception as exc:
            log.exception("stream_failed")
            yield {"event": "error", "data": json.dumps({"message": f"{type(exc).__name__}: {exc}"})}

    return EventSourceResponse(gen(), ping=15)


# ── app ─────────────────────────────────────────────────────────────────────────────────────


def create_app(services: Services | None = None, settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    limiter = RateLimiter(settings.rate_limit_per_minute)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        configure_logging()
        svc = services or await build_services(settings)
        app.state.services = svc
        mcp = svc.mcp_server()
        security = None
        if settings.mcp_allowed_hosts:
            security = TransportSecuritySettings(
                allowed_hosts=settings.mcp_allowed_hosts, allowed_origins=settings.cors_origins
            )
        # Mounted last so every /api route matches first; the MCP app serves exactly /mcp.
        app.mount("/", mcp.streamable_http_app(streamable_http_path="/mcp", transport_security=security))
        async with mcp.session_manager.run():
            yield
        if services is None:
            await svc.aclose()

    app = FastAPI(title="Footnote", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
        expose_headers=["mcp-session-id"],
    )
    app.add_middleware(MCPAuthMiddleware, keys=settings.api_keys)

    def svc(request: Request) -> Services:
        return request.app.state.services

    def principal(request: Request) -> str:
        return _authenticate(settings.api_keys, request.headers.get("authorization", ""))

    async def guard(request: Request, s: Services = Depends(svc), owner_id: str = Depends(principal)) -> Services:
        """Before any route that spends model tokens."""
        limiter.check(owner_id if settings.api_keys else _client_key(request))
        if not s.offline and await s.repo.get_spend() >= settings.daily_budget_usd:
            raise HTTPException(429, "Today's demo budget is used up; it resets at midnight UTC.")
        return s

    @app.get("/api/health")
    async def health(s: Services = Depends(svc)) -> dict[str, Any]:
        return {
            "status": "ok",
            "offline": s.offline,
            "models": {name: s.models.model_for(name) for name in s.config.routes},
            "judge": s.judges.name,
            "corpus": await s.repo.counts(),
        }

    @app.get("/api/scholars")
    async def scholars(s: Services = Depends(svc)) -> list[dict[str, Any]]:
        return [x.model_dump() for x in await s.repo.list_scholars()]

    @app.get("/api/scholars/{scholar_id}")
    async def scholar(scholar_id: str, s: Services = Depends(svc)) -> dict[str, Any]:
        found = await s.repo.get_scholar(scholar_id)
        if found is None:
            raise HTTPException(404, "No such scholar")
        return {
            "scholar": found.model_dump(),
            "documents": [d.model_dump() for d in await s.repo.list_documents(scholar_id)],
        }

    @app.get("/api/documents/{document_id}")
    async def document(document_id: str, s: Services = Depends(svc)) -> dict[str, Any]:
        doc = await s.repo.get_document(document_id)
        if doc is None:
            raise HTTPException(404, "No such document")
        return {
            **doc.model_dump(exclude={"text", "sections", "sha256", "retrieved_at"}),
            "sections": [x.title for x in doc.sections],
        }

    @app.get("/api/passages/{passage_id}")
    async def passage(passage_id: str, s: Services = Depends(svc)) -> dict[str, Any]:
        chunks = await s.repo.get_chunks([passage_id])
        if passage_id not in chunks:
            raise HTTPException(404, "No such passage")
        return Passage.from_chunk(chunks[passage_id]).model_dump()

    @app.post("/api/ask")
    async def ask(body: AskRequest, s: Services = Depends(guard)) -> EventSourceResponse:
        return _sse(
            run_ask(
                s,
                question=body.question,
                scholar_id=body.scholar_id,
                strategy=body.strategy,
                history=[m.model_dump() for m in body.history],
            )
        )

    @app.post("/api/drafts")
    async def start_draft(
        body: DraftRequest, s: Services = Depends(guard), owner_id: str = Depends(principal)
    ) -> EventSourceResponse:
        return _sse(
            run_draft_start(
                s,
                brief=body.brief,
                audience=body.audience,
                scholar_id=body.scholar_id,
                document_id=body.document_id,
                owner_id=owner_id,
            )
        )

    @app.post("/api/drafts/{draft_id}/review")
    async def review(
        draft_id: str,
        body: ReviewRequest,
        s: Services = Depends(guard),
        owner_id: str = Depends(principal),
    ) -> EventSourceResponse:
        draft = await s.repo.get_draft(draft_id, owner_id)
        if draft is None:
            raise HTTPException(404, "No such draft")
        if draft.get("status") != "awaiting_review":
            raise HTTPException(409, f"Draft is {draft.get('status')}, not awaiting review")
        if not await s.repo.claim_draft_review(draft_id, owner_id):
            raise HTTPException(409, "Draft review was already claimed by another request")
        return _sse(run_draft_review(s, draft_id=draft_id, decision=body.decision, notes=body.notes, owner_id=owner_id))

    @app.get("/api/drafts")
    async def drafts(s: Services = Depends(svc), owner_id: str = Depends(principal)) -> list[dict[str, Any]]:
        return json.loads(json.dumps(await s.repo.list_drafts(owner_id=owner_id), default=_json_default))

    @app.get("/api/drafts/{draft_id}")
    async def get_draft(
        draft_id: str, s: Services = Depends(svc), owner_id: str = Depends(principal)
    ) -> dict[str, Any]:
        draft = await s.repo.get_draft(draft_id, owner_id)
        if draft is None:
            raise HTTPException(404, "No such draft")
        return json.loads(json.dumps({k: v for k, v in draft.items() if k != "owner_id"}, default=_json_default))

    @app.post("/api/chat")
    async def chat(
        body: ChatRequest, s: Services = Depends(guard), owner_id: str = Depends(principal)
    ) -> EventSourceResponse:
        return _sse(run_chat(s, [m.model_dump() for m in body.messages], owner_id=owner_id))

    @app.get("/api/evals")
    def evals() -> dict[str, Any]:  # sync: FastAPI runs it in a worker thread (file I/O)
        path: Path = EVALS_LATEST
        if not path.exists():
            return {"generated_at": None, "retrieval": [], "answers": [], "judge": []}
        return json.loads(path.read_text(encoding="utf-8"))

    return app


def main() -> None:
    """`footnote-api`: run the API with uvicorn."""
    import uvicorn

    uvicorn.run("footnote.api.app:create_app", factory=True, host="0.0.0.0", port=8000, proxy_headers=True)
