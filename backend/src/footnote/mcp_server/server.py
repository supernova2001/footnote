"""Footnote as an MCP server.

The same tools serve three kinds of client:
  - Footnote's own agents (in-process; agents/mcp_tools.py),
  - any MCP host over Streamable HTTP at /mcp (mounted in the API),
  - Claude Code / Claude Desktop over stdio:  claude mcp add footnote -- uv run footnote-mcp

Two tools are agents themselves: ask_corpus runs the RAG agent end to end and returns a judged,
cited answer; start_draft runs the writer agent until it pauses for human review.
"""

from __future__ import annotations

import asyncio
from typing import Any, Literal

import mcp.types as mt
from mcp.server.mcpserver import MCPServer

from footnote.auth import current_owner
from footnote.domain import Audience, Strategy
from footnote.services import Services

INSTRUCTIONS = """\
Footnote is a research collection: biographies and public-domain writings of historical Black \
American scholars, writers and educators, plus open-access research papers on reading and literacy. \
Every passage carries its source and licence. Use ask_corpus for questions (answers are \
fact-checked against the passages); search_passages and get_passage to read sources directly."""

READ = mt.ToolAnnotations(read_only_hint=True, open_world_hint=False)


def _passage(p: Any) -> dict[str, Any]:
    return {
        "id": p.id,
        "title": p.title,
        "section": p.section,
        "text": p.text,
        "scholar_ids": p.scholar_ids,
        "license_id": p.license_id,
        "source_url": p.source_url,
        "attribution": p.attribution,
        "score": p.score,
    }


def build_mcp_server(services: Services) -> MCPServer:
    from footnote.runs import collect, run_ask, run_draft_start

    mcp = MCPServer(name="footnote", title="Footnote", instructions=INSTRUCTIONS, version="0.1.0")

    @mcp.tool(annotations=READ)
    async def search_passages(
        query: str, scholar_id: str | None = None, k: int = 6, strategy: Strategy = "hybrid_rerank"
    ) -> list[dict[str, Any]]:
        """Search the collection. Returns up to k passages (id, title, section, text, licence, source URL).

        Args:
            query: What to look for; names and key terms work best.
            scholar_id: Restrict to one scholar (ids from list_scholars).
            k: Number of passages, 1-20.
            strategy: bm25 | dense | hybrid | hybrid_rerank (default; most accurate).
        """
        k = max(1, min(k, 20))
        passages = await services.retriever.search(query, strategy=strategy, k=k, scholar_id=scholar_id)
        return [_passage(p) for p in passages]

    @mcp.tool(annotations=READ)
    async def get_passage(passage_id: str) -> dict[str, Any]:
        """Read one passage in full, with its neighbours' ids for reading on.

        Args:
            passage_id: e.g. "gutenberg-408:0042".
        """
        chunks = await services.repo.get_chunks([passage_id])
        if passage_id not in chunks:
            raise ValueError(f"No passage {passage_id!r}")
        c = chunks[passage_id]
        doc_id, ordinal = c.document_id, c.ordinal
        return {
            **c.model_dump(exclude={"char_start", "char_end"}),
            "previous_id": f"{doc_id}:{ordinal - 1:04d}" if ordinal > 0 else None,
            "next_id": f"{doc_id}:{ordinal + 1:04d}",
        }

    @mcp.tool(annotations=READ)
    async def list_scholars(kind: Literal["historical", "author", "all"] = "historical") -> list[dict[str, Any]]:
        """List the scholars in the collection (historical figures, or authors of the research papers)."""
        scholars = await services.repo.list_scholars()
        return [s.model_dump() for s in scholars if kind in ("all", s.kind)]

    @mcp.tool(annotations=READ)
    async def get_scholar(scholar_id: str) -> dict[str, Any]:
        """One scholar and the documents held for them (title, kind, year, licence, chunk count)."""
        scholar = await services.repo.get_scholar(scholar_id)
        if scholar is None:
            raise ValueError(f"No scholar {scholar_id!r}; see list_scholars")
        docs = await services.repo.list_documents(scholar_id)
        return {"scholar": scholar.model_dump(), "documents": [d.model_dump() for d in docs]}

    @mcp.tool(annotations=READ)
    async def list_documents(scholar_id: str | None = None) -> list[dict[str, Any]]:
        """List documents, optionally for one scholar."""
        return [d.model_dump() for d in await services.repo.list_documents(scholar_id)]

    @mcp.tool(annotations=mt.ToolAnnotations(read_only_hint=True, open_world_hint=False))
    async def ask_corpus(question: str, scholar_id: str | None = None) -> dict[str, Any]:
        """Ask a question; returns a fact-checked answer with numbered citations [n] and its sources.

        Runs Footnote's RAG agent: query planning, hybrid retrieval with reranking, a cited answer,
        and a grounding judge on every claim. Claims without full support and an explicit citation
        are withheld and listed under `withheld`.
        """
        result = await collect(run_ask(services, question=question, scholar_id=scholar_id, strategy="hybrid_rerank"))
        answer = result.get("answer", {})
        passages = {p["id"]: p for p in result.get("passages", {}).get("passages", [])}
        numbering: dict[str, int] = {}
        parts = []
        for seg in answer.get("segments", []):
            marks = "".join(f"[{numbering.setdefault(c['passage_id'], len(numbering) + 1)}]" for c in seg["citations"])
            parts.append(seg["text"].rstrip() + marks + " ")
        sources = [
            {
                "n": n,
                "passage_id": pid,
                "title": passages.get(pid, {}).get("title"),
                "source_url": passages.get(pid, {}).get("source_url"),
                "license_id": passages.get(pid, {}).get("license_id"),
            }
            for pid, n in numbering.items()
        ]
        return {
            "answer": "".join(parts).strip(),
            "sources": sources,
            "withheld": [s["text"] for s in answer.get("removed", [])],
            "abstained": answer.get("abstained", False),
            "run_id": result.get("done", {}).get("run_id"),
        }

    @mcp.tool(
        annotations=mt.ToolAnnotations(
            read_only_hint=False, destructive_hint=False, idempotent_hint=False, open_world_hint=False
        )
    )
    async def start_draft(
        brief: str, audience: Audience = "grade8", scholar_id: str | None = None, document_id: str | None = None
    ) -> dict[str, Any]:
        """Write a cited article from the collection. Stops at human review; returns the draft id.

        Runs Footnote's writer agent: a research agent picks passages, sections are drafted in
        parallel, each claim is judged and failing ones redrafted, and the draft waits for a person
        to approve it on the Write page.

        Args:
            brief: What the article should cover.
            audience: grade4 | grade8 | adult.
        """
        result = await collect(
            run_draft_start(
                services,
                brief=brief,
                audience=audience,
                scholar_id=scholar_id,
                document_id=document_id,
                owner_id=current_owner.get(),
            )
        )
        if "error" in result:
            raise RuntimeError(result["error"]["message"])
        draft = result.get("review", {}).get("draft", {})
        return {
            "draft_id": result.get("draft", {}).get("draft_id"),
            "title": draft.get("title"),
            "status": draft.get("status"),
            "sections": [s["heading"] for s in draft.get("sections", [])],
        }

    @mcp.resource("footnote://scholars/{scholar_id}", mime_type="text/markdown")
    async def scholar_resource(scholar_id: str) -> str:
        """A scholar's summary and document list, as Markdown."""
        scholar = await services.repo.get_scholar(scholar_id)
        if scholar is None:
            raise ValueError(f"No scholar {scholar_id!r}")
        docs = await services.repo.list_documents(scholar_id)
        life = f" ({scholar.born or '?'}–{scholar.died or ''})" if scholar.born or scholar.died else ""
        lines = [f"# {scholar.name}{life}", "", scholar.summary, "", "## Documents"]
        lines += [f"- {d.title} ({d.year or 'n.d.'}): {d.license.attribution}" for d in docs]
        return "\n".join(lines)

    @mcp.resource("footnote://documents/{document_id}", mime_type="text/plain")
    async def document_resource(document_id: str) -> str:
        """A document's full text with its licence attribution."""
        doc = await services.repo.get_document(document_id)
        if doc is None:
            raise ValueError(f"No document {document_id!r}")
        return f"{doc.title}\n{doc.license.attribution}\n\n{doc.text}"

    @mcp.prompt()
    def write_story(topic: str, audience: str = "grade8") -> str:
        """Ask the assistant to research and draft a sourced story with Footnote's tools."""
        return (
            f"Using Footnote's tools, research {topic!r} in the collection and write a short story for "
            f"{audience} readers. Use ask_corpus for facts, quote only from get_passage, and cite every "
            f"factual sentence. If you want a reviewed article saved, call start_draft."
        )

    return mcp


def main() -> None:
    """`footnote-mcp`: serve over stdio for Claude Code / Claude Desktop."""
    import sys

    from footnote.logs import configure_logging
    from footnote.services import build_services
    from footnote.settings import get_settings

    configure_logging(stream=sys.stderr)  # stdout carries the MCP protocol

    async def serve() -> None:
        services = await build_services(get_settings())
        try:
            await build_mcp_server(services).run_stdio_async()
        finally:
            await services.aclose()

    asyncio.run(serve())
