"""Run the agent graphs and turn their streams into the contract's events.

Shared by the HTTP API (as server-sent events) and the MCP tools (consumed to completion), so a
question asked from the web page and one asked by Claude Code through MCP take the same path:
the same graph, the same cost ledger, the same run record.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

import structlog
from langgraph.types import Command

from footnote.agents.ask import answer_payload, initial_state
from footnote.agents.writer import draft_document
from footnote.auth import current_owner
from footnote.domain import Audience, Strategy
from footnote.llm.client import start_ledger
from footnote.services import Services

log = structlog.get_logger(__name__)
Event = tuple[str, dict[str, Any]]


async def stream_graph(graph: Any, graph_input: Any, config: dict[str, Any]) -> AsyncIterator[Event]:
    async for mode, chunk in graph.astream(graph_input, config, stream_mode=["custom", "updates", "values"]):
        if mode == "custom":
            yield chunk["event"], chunk["data"]
        elif mode == "updates":
            for node in chunk:
                if node == "__interrupt__":
                    yield "__interrupt__", {}
                else:
                    yield "step", {"node": node, "status": "done", "detail": None}
        elif mode == "values":
            yield "__values__", chunk


def _done(run_id: uuid.UUID, ledger: Any, started: float) -> dict[str, Any]:
    return {
        "run_id": str(run_id),
        "usage": ledger.usage.model_dump(),
        "latency_ms": int((time.perf_counter() - started) * 1000),
        "trace_url": None,  # the LangSmith run id equals run_id; see README "Tracing"
    }


async def _persist_run(
    services: Services,
    *,
    kind: str,
    status: str,
    run_id: uuid.UUID,
    ledger: Any,
    started: float,
    error: str | None = None,
    **fields: Any,
) -> dict[str, Any]:
    """Persist terminal telemetry without hiding the run's actual result.

    Both writes are attempted independently. This function is called from each runner's `finally`
    block, including cancellation and generator close paths.
    """
    done = _done(run_id, ledger, started)
    record = {"kind": kind, "status": status, "calls": ledger.calls, **fields, **done}
    if error:
        record["error"] = error
    results = await asyncio.gather(
        services.repo.record_run(record),
        services.repo.add_spend(ledger.usage.cost_usd),
        return_exceptions=True,
    )
    for operation, result in zip(("record_run", "add_spend"), results, strict=True):
        if isinstance(result, BaseException):
            log.error("run_persistence_failed", operation=operation, kind=kind, run_id=str(run_id), error=str(result))
    return done


async def run_ask(
    services: Services,
    *,
    question: str,
    scholar_id: str | None,
    strategy: Strategy,
    history: list[dict[str, str]] | None = None,
) -> AsyncIterator[Event]:
    ledger, started, run_id = start_ledger(), time.perf_counter(), uuid.uuid4()
    status, error = "cancelled", None
    payload: dict[str, Any] | None = None
    done: dict[str, Any]
    try:
        scholar = await services.repo.get_scholar(scholar_id) if scholar_id else None
        config = {
            "run_id": run_id,
            "run_name": "ask",
            "tags": ["ask", strategy],
            "metadata": {"scholar_id": scholar_id, "strategy": strategy},
        }
        final: dict[str, Any] = {}
        state = initial_state(question, strategy, scholar_id, scholar.name if scholar else None, history or [])
        async for event, data in stream_graph(services.ask_graph, state, config):
            if event == "__values__":
                final = data
                continue
            yield event, data
        payload = answer_payload(final)  # type: ignore[arg-type]
        status = "completed"
        yield "answer", payload
    except asyncio.CancelledError:
        raise
    except GeneratorExit:
        raise
    except Exception as exc:
        status, error = "failed", str(exc)
        raise
    finally:
        done = await _persist_run(
            services,
            kind="ask",
            status=status,
            run_id=run_id,
            ledger=ledger,
            started=started,
            error=error,
            question=question,
            strategy=strategy,
            scholar_id=scholar_id,
            abstained=payload["abstained"] if payload else None,
        )
    yield "done", done


async def _stream_writer(
    services: Services, graph_input: Any, draft_id: str, run_id: uuid.UUID
) -> AsyncIterator[Event]:
    config = {
        "configurable": {"thread_id": draft_id},
        "run_id": run_id,
        "run_name": "writer",
        "tags": ["writer"],
        "metadata": {"draft_id": draft_id},
    }
    final: dict[str, Any] = {}
    interrupted = False
    try:
        async for event, data in stream_graph(services.writer_graph, graph_input, config):
            if event == "__values__":
                final = data
            elif event == "__interrupt__":
                interrupted = True
            else:
                yield event, data
    except Exception as exc:
        log.exception("writer_failed", draft_id=draft_id)
        if final:
            await services.repo.save_draft(draft_document({**final, "status": "failed"}))
        yield "error", {"message": str(exc)}
        return
    draft = await services.repo.get_draft(draft_id) or draft_document(final)
    if interrupted:
        yield "review", {"draft": draft}
    elif final.get("status") == "published":
        yield "published", {"draft": draft}


async def run_draft_start(
    services: Services,
    *,
    brief: str,
    audience: Audience,
    scholar_id: str | None,
    document_id: str | None,
    owner_id: str = "anonymous",
) -> AsyncIterator[Event]:
    ledger, started, run_id = start_ledger(), time.perf_counter(), uuid.uuid4()
    draft_id = f"d_{uuid.uuid4().hex[:12]}"
    status, error = "cancelled", None
    done: dict[str, Any]
    try:
        yield "draft", {"draft_id": draft_id}
        state: dict[str, Any] = {
            "draft_id": draft_id,
            "owner_id": owner_id,
            "brief": brief,
            "audience": audience,
            "scholar_id": scholar_id,
            "document_id": document_id,
            "status": "running",
            "sections": [],
        }
        await services.repo.save_draft(draft_document(state))
        status = "completed"
        async for event, data in _stream_writer(services, state, draft_id, run_id):
            if event == "error":
                status, error = "failed", str(data.get("message", "writer failed"))
            elif event == "review":
                status = "interrupted"
            yield event, data
    except asyncio.CancelledError:
        status = "cancelled"
        raise
    except GeneratorExit:
        status = "cancelled"
        raise
    except Exception as exc:
        status, error = "failed", str(exc)
        raise
    finally:
        done = await _persist_run(
            services,
            kind="draft_start",
            status=status,
            run_id=run_id,
            ledger=ledger,
            started=started,
            error=error,
            draft_id=draft_id,
            brief=brief,
            audience=audience,
            scholar_id=scholar_id,
            document_id=document_id,
            owner_id=owner_id,
        )
    yield "done", done


async def run_draft_review(
    services: Services, *, draft_id: str, decision: str, notes: str | None, owner_id: str = "anonymous"
) -> AsyncIterator[Event]:
    ledger, started, run_id = start_ledger(), time.perf_counter(), uuid.uuid4()
    status, error = "cancelled", None
    done: dict[str, Any]
    try:
        status = "completed"
        async for event, data in _stream_writer(
            services, Command(resume={"decision": decision, "notes": notes}), draft_id, run_id
        ):
            if event == "error":
                status, error = "failed", str(data.get("message", "writer failed"))
            elif event == "review":
                status = "interrupted"
            yield event, data
    except asyncio.CancelledError:
        status = "cancelled"
        raise
    except GeneratorExit:
        status = "cancelled"
        raise
    except Exception as exc:
        status, error = "failed", str(exc)
        raise
    finally:
        if status in {"cancelled", "failed"}:
            await services.repo.release_draft_review(draft_id, owner_id)
        done = await _persist_run(
            services,
            kind="draft_review",
            status=status,
            run_id=run_id,
            ledger=ledger,
            started=started,
            error=error,
            draft_id=draft_id,
            decision=decision,
            owner_id=owner_id,
        )
    yield "done", done


async def run_chat(
    services: Services, messages: list[dict[str, str]], owner_id: str = "anonymous"
) -> AsyncIterator[Event]:
    ledger, started, run_id = start_ledger(), time.perf_counter(), uuid.uuid4()
    status, error = "cancelled", None
    done: dict[str, Any]
    owner_token = current_owner.set(owner_id)
    try:
        status = "completed"
        async for event, data in services.supervisor.chat(messages):
            if event == "error":
                status, error = "failed", str(data.get("message", "supervisor failed"))
            yield event, data
    except asyncio.CancelledError:
        status = "cancelled"
        raise
    except GeneratorExit:
        status = "cancelled"
        raise
    except Exception as exc:
        status, error = "failed", str(exc)
        raise
    finally:
        current_owner.reset(owner_token)
        done = await _persist_run(
            services,
            kind="chat",
            status=status,
            run_id=run_id,
            ledger=ledger,
            started=started,
            error=error,
            message_count=len(messages),
            owner_id=owner_id,
        )
    yield "done", done


async def collect(events: AsyncIterator[Event]) -> dict[str, dict[str, Any]]:
    """Consume a run to completion; keep the last payload of each event type (for MCP tools)."""
    last: dict[str, dict[str, Any]] = {}
    async for event, data in events:
        if event not in ("token", "step", "section"):
            last[event] = data
    return last
