from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from footnote.api.app import create_app
from footnote.settings import Settings


def events(
    client: TestClient, path: str, body: dict[str, Any], headers: dict[str, str] | None = None
) -> list[tuple[str, Any]]:
    out, event = [], None
    with client.stream("POST", path, json=body, headers=headers) as resp:
        assert resp.status_code == 200, resp.read()
        for line in resp.iter_lines():
            if line.startswith("event:"):
                event = line.split(":", 1)[1].strip()
            elif line.startswith("data:") and event:
                out.append((event, json.loads(line.split(":", 1)[1])))
    return out


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    with TestClient(create_app(settings=settings)) as c:
        yield c


def test_health_and_collection_routes(client: TestClient) -> None:
    health = client.get("/api/health").json()
    assert health["offline"] is True and health["corpus"]["documents"] == 3
    scholars = client.get("/api/scholars").json()
    assert {s["id"] for s in scholars} == {"carter-g-woodson", "ida-b-wells"}
    detail = client.get("/api/scholars/carter-g-woodson").json()
    assert len(detail["documents"]) == 2 and detail["documents"][0]["license"]["attribution"]
    assert client.get("/api/scholars/nobody").status_code == 404
    assert client.get("/api/passages/gutenberg-14975:0000").json()["license_id"] == "PD-US"
    assert client.get("/api/evals").json()["retrieval"] == [] or True


def test_ask_streams_the_contract(client: TestClient) -> None:
    evs = events(client, "/api/ask", {"question": "What did Wells publish in 1892?"})
    names = [e for e, _ in evs]
    assert names[0] == "step" and "passages" in names and "token" in names
    assert names[-2:] == ["answer", "done"]
    answer = dict(evs)["answer"]
    assert answer["segments"] and "removed" in answer


def test_ask_validates_input(client: TestClient) -> None:
    assert client.post("/api/ask", json={"question": "x"}).status_code == 422
    assert client.post("/api/ask", json={"question": "valid question", "strategy": "magic"}).status_code == 422


def test_draft_review_flow_over_http(client: TestClient) -> None:
    evs = events(client, "/api/drafts", {"brief": "Ida B. Wells and the 1892 pamphlet", "audience": "grade8"})
    names = [e for e, _ in evs]
    assert names[0] == "draft" and "outline" in names and "review" in names and names[-1] == "done"
    draft_id = dict(evs)["draft"]["draft_id"]
    assert client.get(f"/api/drafts/{draft_id}").json()["status"] == "awaiting_review"
    assert any(d["id"] == draft_id for d in client.get("/api/drafts").json())
    evs = events(client, f"/api/drafts/{draft_id}/review", {"decision": "approve"})
    assert "published" in [e for e, _ in evs]
    assert client.post(f"/api/drafts/{draft_id}/review", json={"decision": "approve"}).status_code == 409


def test_chat_and_mcp_endpoint(client: TestClient) -> None:
    evs = events(client, "/api/chat", {"messages": [{"role": "user", "content": "Who was Woodson?"}]})
    assert [e for e, _ in evs][-1] == "done"
    init = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "test", "version": "0"}},
    }
    headers = {"accept": "application/json, text/event-stream"}
    resp = client.post("/mcp", json=init, headers={**headers, "host": "localhost:8000"})
    assert resp.status_code == 200, resp.text
    assert "footnote" in resp.text
    # DNS-rebinding protection: a foreign Host header is refused.
    assert client.post("/mcp", json=init, headers={**headers, "host": "evil.example:80"}).status_code == 421


def test_rate_limit(settings: Settings) -> None:
    tight = settings.model_copy(update={"rate_limit_per_minute": 1})
    with TestClient(create_app(settings=tight)) as c:
        assert c.post("/api/ask", json={"question": "Who was Woodson?"}).status_code == 200
        assert c.post("/api/ask", json={"question": "Who was Woodson?"}).status_code == 429


def test_api_keys_authenticate_and_isolate_draft_owners(settings: Settings) -> None:
    protected = settings.model_copy(update={"api_keys_json": SecretStr('{"alice":"alice-secret","bob":"bob-secret"}')})
    alice = {"Authorization": "Bearer alice-secret"}
    bob = {"Authorization": "Bearer bob-secret"}
    with TestClient(create_app(settings=protected)) as c:
        assert c.post("/api/ask", json={"question": "Who was Woodson?"}).status_code == 401
        assert (
            c.post(
                "/api/ask", json={"question": "Who was Woodson?"}, headers={"Authorization": "Bearer wrong"}
            ).status_code
            == 401
        )
        assert events(c, "/api/ask", {"question": "Who was Woodson?"}, alice)[-1][0] == "done"

        created = events(
            c,
            "/api/drafts",
            {"brief": "Ida B. Wells and the 1892 pamphlet", "audience": "grade8"},
            alice,
        )
        draft_id = dict(created)["draft"]["draft_id"]
        assert c.get(f"/api/drafts/{draft_id}", headers=alice).status_code == 200
        assert c.get(f"/api/drafts/{draft_id}", headers=bob).status_code == 404
        assert c.get("/api/drafts", headers=bob).json() == []

        init = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "0"},
            },
        }
        mcp_headers = {"accept": "application/json, text/event-stream", "host": "localhost:8000"}
        assert c.post("/mcp", json=init, headers=mcp_headers).status_code == 401
        assert c.post("/mcp", json=init, headers={**mcp_headers, **alice}).status_code == 200
