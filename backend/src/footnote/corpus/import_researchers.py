"""Import a researcher-profile export into Footnote's licensed corpus contract.

The source collections and source Pinecone index are never modified. Each accepted source
passage becomes one Footnote document and is re-chunked from repaired text so character offsets
are derived from, and traceable to, the exact text stored in MongoDB.
"""

from __future__ import annotations

import asyncio
import hashlib
import html
import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse

import httpx

from footnote.corpus.chunking import chunk_document
from footnote.corpus.licenses import make_license
from footnote.db.repo import MongoRepo
from footnote.domain import Document, LicenseId, Scholar
from footnote.retrieval.store import PineconeStore
from footnote.settings import Settings

_DOI = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.I)
_ARXIV = re.compile(r"arxiv\.org/(?:pdf|abs)/(?:[a-z-]+/)?([0-9]{4}\.[0-9]{4,5}|[a-z-]+/[0-9]{7})", re.I)
_PMC = re.compile(r"PMC\d+", re.I)
_SPACE = re.compile(r"[ \t]+")
_BLANKS = re.compile(r"\n{3,}")
_MOJIBAKE = {
    "â€œ": "“",
    "â€": "”",
    "â€™": "’",
    "â€˜": "‘",
    "â€“": "–",
    "â€”": "—",
    "â€¦": "…",
    "Â©": "©",
    "Â®": "®",
    "Â ": " ",
    "�": "",
    "": "’",
    "": "“",
    "": "”",
}


@dataclass(frozen=True)
class LicenseDecision:
    license_id: LicenseId | None
    evidence_url: str | None
    reason: str


@dataclass
class MigrationReport:
    source_scholars: int = 0
    source_passages: int = 0
    accepted_documents: int = 0
    accepted_scholars: int = 0
    chunks: int = 0
    indexed: int = 0
    indexed_this_run: int = 0
    skipped_by_reason: Counter[str] = field(default_factory=Counter)
    license_counts: Counter[str] = field(default_factory=Counter)
    repairs: Counter[str] = field(default_factory=Counter)
    skipped: list[dict[str, str]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_scholars": self.source_scholars,
            "source_passages": self.source_passages,
            "accepted_documents": self.accepted_documents,
            "accepted_scholars": self.accepted_scholars,
            "chunks": self.chunks,
            "indexed": self.indexed,
            "indexed_this_run": self.indexed_this_run,
            "skipped_by_reason": dict(self.skipped_by_reason),
            "license_counts": dict(self.license_counts),
            "repairs": dict(self.repairs),
            "skipped": self.skipped,
        }


def _repair_mojibake(text: str) -> tuple[str, int]:
    count = 0
    for broken, fixed in _MOJIBAKE.items():
        n = text.count(broken)
        if n:
            text, count = text.replace(broken, fixed), count + n
    return text, count


def repair_text(raw: str) -> tuple[str, Counter[str]]:
    """Conservatively repair encoding/control noise and discard obvious PDF extraction runs."""
    repairs: Counter[str] = Counter()
    text = html.unescape(raw).replace("\r\n", "\n").replace("\r", "\n")
    text, n = _repair_mojibake(text)
    repairs["mojibake_sequences"] += n

    cleaned_chars: list[str] = []
    for char in text:
        if unicodedata.category(char) == "Cc" and char not in "\n\t":
            cleaned_chars.append(" ")
            repairs["control_characters"] += 1
        else:
            cleaned_chars.append(char)
    text = "".join(cleaned_chars)

    lines: list[str] = []
    for raw_line in text.splitlines():
        line = _SPACE.sub(" ", raw_line).strip()
        tokens = re.findall(r"[A-Za-z0-9]+", line)
        if len(tokens) >= 20:
            token, count = Counter(t.lower() for t in tokens).most_common(1)[0]
            if len(token) == 1 and count / len(tokens) >= 0.70:
                repairs["repetitive_pdf_lines"] += 1
                continue
        lines.append(line)
    text = _BLANKS.sub("\n\n", "\n".join(lines)).strip()
    return unicodedata.normalize("NFC", text), repairs


def _license_in_text(text: str) -> LicenseId | None:
    lower = html.unescape(text).lower().replace("http://", "https://")
    if "creativecommons.org/licenses/by-sa/4.0" in lower:
        return "CC-BY-SA-4.0"
    if "creativecommons.org/publicdomain/zero/1.0" in lower:
        return "CC0-1.0"
    if "creativecommons.org/licenses/by/4.0" in lower:
        return "CC-BY-4.0"
    if re.search(r"\b(?:this (?:work|article)|the work) is (?:in )?the public domain\b", lower):
        return "PD-US"
    return None


def _extract_doi(url: str) -> str | None:
    match = _DOI.search(url)
    if not match:
        return None
    doi = match.group(0).rstrip(".,);]}")
    return re.sub(r"/(?:pdf|full|abstract)$", "", doi, flags=re.I)


def _landing_url(url: str) -> str:
    parsed = urlparse(url)
    if not parsed.scheme and _extract_doi(url):
        return f"https://doi.org/{url}"
    if "journals.plos.org" in parsed.netloc and "/article/file" in parsed.path:
        return url.replace("/article/file", "/article").split("&type=", 1)[0]
    if "frontiersin.org" in parsed.netloc and parsed.path.endswith("/pdf"):
        return url.removesuffix("/pdf")
    if "biorxiv.org" in parsed.netloc:
        return re.sub(r"\.full\.pdf(?:\?.*)?$", "", url)
    return url


class LicenseVerifier:
    """Accept only an explicit supported licence found in source or registry metadata."""

    def __init__(
        self, client: httpx.AsyncClient, cache_path: Path, workers: int = 16, *, retry_unresolved: bool = True
    ) -> None:
        self.client = client
        self.cache_path = cache_path
        self.semaphore = asyncio.Semaphore(workers)
        self.retry_unresolved = retry_unresolved
        if cache_path.exists():
            self.cache: dict[str, dict[str, Any]] = json.loads(cache_path.read_text(encoding="utf-8"))
        else:
            self.cache = {}

    async def verify(self, source_url: str) -> LicenseDecision:
        if source_url in self.cache:
            row = self.cache[source_url]
            # Successful evidence is stable. Negative results are retried because transient registry
            # throttling must not permanently turn a verifiable source into an exclusion.
            if row.get("license_id") or not self.retry_unresolved:
                return LicenseDecision(row["license_id"], row.get("evidence_url"), row["reason"])
        async with self.semaphore:
            decision = await self._verify_uncached(source_url)
        self.cache[source_url] = {
            "license_id": decision.license_id,
            "evidence_url": decision.evidence_url,
            "reason": decision.reason,
            "checked_at": datetime.now(UTC).isoformat(),
        }
        return decision

    async def _get(self, url: str) -> tuple[str, str] | None:
        for attempt in range(3):
            try:
                response = await self.client.get(url, follow_redirects=True, timeout=20)
                if response.status_code == 429 and attempt < 2:
                    await asyncio.sleep(min(float(response.headers.get("retry-after", "2")), 10))
                    continue
                response.raise_for_status()
                content_type = response.headers.get("content-type", "")
                if "pdf" in content_type or len(response.content) > 5_000_000:
                    return None
                return str(response.url), response.text
            except (httpx.HTTPError, UnicodeError, ValueError):
                return None
        return None

    async def _verify_uncached(self, source_url: str) -> LicenseDecision:
        doi = _extract_doi(source_url)
        if doi:
            endpoint = f"https://api.crossref.org/works/{quote(doi, safe='')}"
            result = await self._get(endpoint)
            if result:
                try:
                    message = json.loads(result[1])["message"]
                    for item in message.get("license", []):
                        if found := _license_in_text(str(item.get("URL", ""))):
                            return LicenseDecision(found, endpoint, "explicit Crossref licence URL")
                except (KeyError, TypeError, json.JSONDecodeError):
                    pass

        pmc = _PMC.search(source_url)
        if pmc:
            endpoint = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmc.group(0).upper()}/fullTextXML"
            result = await self._get(endpoint)
            if result and (found := _license_in_text(result[1])):
                return LicenseDecision(found, endpoint, "explicit licence in Europe PMC article XML")

        arxiv = _ARXIV.search(source_url)
        if arxiv:
            endpoint = f"https://export.arxiv.org/api/query?id_list={quote(arxiv.group(1), safe='')}"
            result = await self._get(endpoint)
            if result and (found := _license_in_text(result[1])):
                return LicenseDecision(found, endpoint, "explicit arXiv record licence")

        result = await self._get(_landing_url(source_url))
        if result and (found := _license_in_text(result[1])):
            return LicenseDecision(found, result[0], "explicit licence on source page")
        return LicenseDecision(None, None, "no explicit accepted licence found")

    def save(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self.cache, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _source_url(passage: dict[str, Any]) -> str:
    sources = passage.get("sources") or []
    return str(sources[0].get("url", "")).strip() if sources else ""


def _normalise_source_url(source_url: str) -> str:
    if not urlparse(source_url).scheme and _extract_doi(source_url):
        return f"https://doi.org/{source_url}"
    return source_url


def _attribution(title: str, scholar: str, year: int | None, license_id: LicenseId, source_url: str) -> str:
    date = str(year) if year else "n.d."
    return f"“{title}” by {scholar} ({date}). {license_id}. Source: {source_url}"


async def import_researchers(
    settings: Settings,
    *,
    source_db: str,
    apply: bool,
    cache_path: Path,
    report_path: Path,
    workers: int = 16,
) -> MigrationReport:
    from pymongo import AsyncMongoClient

    client: AsyncMongoClient[dict[str, Any]] = AsyncMongoClient(
        settings.mongodb_uri.get_secret_value(), appname="footnote-researcher-import"
    )
    source = client[source_db]
    report = MigrationReport()
    try:
        source_scholars = {str(d["_id"]): d async for d in source.scholars.find({})}
        passages = [d async for d in source.passages.find({})]
        report.source_scholars, report.source_passages = len(source_scholars), len(passages)

        headers = {"User-Agent": settings.user_agent, "Accept": "application/json, application/xml, text/html"}
        async with httpx.AsyncClient(headers=headers) as http:
            verifier = LicenseVerifier(http, cache_path, workers, retry_unresolved=not apply)
            urls = sorted({_source_url(p) for p in passages if _source_url(p)})
            results = await asyncio.gather(*(verifier.verify(url) for url in urls))
            decisions = dict(zip(urls, results, strict=True))
            await asyncio.to_thread(verifier.save)

        documents: list[Document] = []
        document_scholars: Counter[str] = Counter()
        for passage in passages:
            pid = str(passage["_id"])
            sid = str(passage.get("scholar_id", ""))
            verification_url = _source_url(passage)
            source_url = _normalise_source_url(verification_url)
            decision = decisions.get(verification_url, LicenseDecision(None, None, "missing source URL"))
            reason = "missing source scholar" if sid not in source_scholars else ""
            if not reason and decision.license_id is None:
                reason = decision.reason
            if reason:
                report.skipped_by_reason[reason] += 1
                report.skipped.append({"id": pid, "source_url": source_url, "reason": reason})
                continue

            text, repairs = repair_text(str(passage.get("text", "")))
            report.repairs.update(repairs)
            if len(re.findall(r"[A-Za-z]{2,}", text)) < 40:
                reason = "too little usable prose after repair"
                report.skipped_by_reason[reason] += 1
                report.skipped.append({"id": pid, "source_url": source_url, "reason": reason})
                continue

            scholar = source_scholars[sid]
            title, name = str(passage.get("title") or "Untitled source"), str(scholar.get("name") or "Unknown")
            title, fixed = _repair_mojibake(title)
            report.repairs["mojibake_sequences"] += fixed
            year = passage.get("year") if isinstance(passage.get("year"), int) else None
            license_id = decision.license_id
            assert license_id is not None
            attribution = _attribution(title, name, year, license_id, source_url)
            document = Document(
                id=pid,
                title=title,
                kind="paper",
                year=year,
                scholar_ids=[sid],
                source_url=source_url,
                license=make_license(license_id, attribution),
                sha256=hashlib.sha256(text.encode()).hexdigest(),
                text=text,
            )
            chunks = chunk_document(document)
            if not chunks:
                reason = "chunker produced no citable passages"
                report.skipped_by_reason[reason] += 1
                report.skipped.append({"id": pid, "source_url": source_url, "reason": reason})
                continue
            document.chunk_count = len(chunks)
            documents.append(document)
            document_scholars[sid] += 1
            report.license_counts[license_id] += 1
            report.chunks += len(chunks)

        report.accepted_documents = len(documents)
        report.accepted_scholars = len(document_scholars)

        if apply:
            target = MongoRepo(client, settings.mongodb_db)
            await target.ensure_indexes()
            if settings.pinecone_api_key is None:
                raise RuntimeError("PINECONE_API_KEY is required for an applied migration")
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
            try:
                await store.ensure_index()
                for sid, count in document_scholars.items():
                    row = source_scholars[sid]
                    await target.upsert_scholar(
                        Scholar(
                            id=sid,
                            name=str(row.get("name") or row.get("display_name") or sid),
                            # Living researchers appear only as the bylines on their papers. The export's
                            # profile bios are model-written summaries and are not republished.
                            summary="",
                            kind="author",
                            document_count=count,
                        )
                    )
                index_batch = []
                for document in documents:
                    chunks = chunk_document(document)
                    existing = await target.get_document(document.id)
                    if existing is not None:
                        document.retrieved_at = existing.retrieved_at
                    needs_index = (
                        existing is None
                        or existing.sha256 != document.sha256
                        or existing.source_url != document.source_url
                        or existing.license != document.license
                    )
                    if existing is not None and existing.sha256 != document.sha256:
                        await store.delete_document(document.id)
                    if existing != document:
                        await target.upsert_document(document)
                        await target.replace_chunks(document.id, chunks)
                    if needs_index:
                        index_batch.extend(chunks)
                    if len(index_batch) >= 480:
                        report.indexed_this_run += await store.upsert(index_batch)
                        index_batch.clear()
                if index_batch:
                    report.indexed_this_run += await store.upsert(index_batch)
                report.indexed = report.chunks
            finally:
                await store.close()
    finally:
        await client.close()

    await asyncio.to_thread(_write_report, report_path, report.as_dict())
    return report


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
