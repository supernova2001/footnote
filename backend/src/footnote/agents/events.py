"""Events a graph node sends to the client while it runs (LangGraph's custom stream mode)."""

from __future__ import annotations

from typing import Any

from langgraph.config import get_stream_writer


def emit(event: str, data: dict[str, Any]) -> None:
    """Send one server-sent event to whoever is streaming this run; a no-op when nobody is."""
    try:
        writer = get_stream_writer()
    except RuntimeError:  # called outside a graph run (a unit test calling a node directly)
        return
    writer({"event": event, "data": data})


def step(node: str, status: str = "start", detail: str | None = None) -> None:
    emit("step", {"node": node, "status": status, "detail": detail})
