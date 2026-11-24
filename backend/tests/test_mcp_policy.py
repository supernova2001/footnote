from __future__ import annotations

import json
from pathlib import Path

import pytest

from footnote.agents.mcp_tools import load_external_servers


def test_enabled_external_server_requires_explicit_tool_allowlist(tmp_path: Path) -> None:
    config = tmp_path / "mcp.json"
    config.write_text(json.dumps({"servers": {"unsafe": {"enabled": True, "command": "example"}}}))
    with pytest.raises(ValueError, match="allowed_tools"):
        load_external_servers(config)


def test_external_server_loads_only_with_allowlist(tmp_path: Path) -> None:
    config = tmp_path / "mcp.json"
    config.write_text(
        json.dumps({"servers": {"safe": {"enabled": True, "command": "example", "allowed_tools": ["lookup"]}}})
    )
    specs = load_external_servers(config)
    assert len(specs) == 1 and specs[0].allowed_tools == ["lookup"]
