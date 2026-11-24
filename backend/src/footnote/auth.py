"""Request identity shared with nested in-process MCP agent calls."""

from __future__ import annotations

from contextvars import ContextVar

current_owner: ContextVar[str] = ContextVar("footnote_owner", default="mcp")
