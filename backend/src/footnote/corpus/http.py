"""A polite HTTP fetcher for the corpus sources.

- An honest User-Agent naming the project (and the operator's contact, if they set one).
- One request at a time per host, with a minimum interval between them.
- robots.txt is honoured for plain page/file fetches. Documented APIs (MediaWiki, Wikidata,
  Europe PMC) are used under their API terms instead; they are listed in API_HOSTS.
- Retries with backoff on 429 / 5xx, respecting Retry-After.
- Every response body is cached on disk, so re-running the ingest does not re-download a book.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urlsplit
from urllib.robotparser import RobotFileParser

import httpx
import structlog

log = structlog.get_logger(__name__)

API_HOSTS = frozenset({"en.wikipedia.org", "www.wikidata.org", "www.ebi.ac.uk"})
# Wikimedia's User-Agent policy (https://meta.wikimedia.org/wiki/User-Agent_policy) requires a way
# to contact the operator, and its servers refuse automated requests without one.
CONTACT_REQUIRED_HOSTS = frozenset({"en.wikipedia.org", "www.wikidata.org"})


class RobotsDisallowed(RuntimeError):
    pass


class ContactRequired(RuntimeError):
    pass


class Fetcher:
    def __init__(
        self, *, user_agent: str, cache_dir: Path, has_contact: bool, min_interval_s: float = 1.0, timeout_s: float = 60
    ) -> None:
        self.user_agent = user_agent
        self.cache_dir = cache_dir
        self.has_contact = has_contact
        self.min_interval_s = min_interval_s
        self._client = httpx.AsyncClient(headers={"User-Agent": user_agent}, timeout=timeout_s, follow_redirects=True)
        self._host_locks: dict[str, asyncio.Lock] = {}
        self._last: dict[str, float] = {}
        self._robots: dict[str, RobotFileParser] = {}

    async def close(self) -> None:
        await self._client.aclose()

    def _cache_path(self, url: str) -> Path:
        digest = hashlib.sha256(url.encode()).hexdigest()[:24]
        return self.cache_dir / urlsplit(url).netloc / digest

    async def _allowed(self, url: str) -> bool:
        parts = urlsplit(url)
        if parts.netloc in API_HOSTS:
            return True
        if parts.netloc not in self._robots:
            rp = RobotFileParser()
            robots_url = f"{parts.scheme}://{parts.netloc}/robots.txt"
            try:
                resp = await self._client.get(robots_url)
                rp.parse(resp.text.splitlines() if resp.status_code == 200 else [])
            except httpx.HTTPError:
                rp.parse([])  # unreachable robots.txt: no rules published
            self._robots[parts.netloc] = rp
        return self._robots[parts.netloc].can_fetch(self.user_agent, url)

    async def get_bytes(self, url: str, params: dict[str, Any] | None = None, *, use_cache: bool = True) -> bytes:
        full = f"{url}?{urlencode(params, doseq=True)}" if params else url
        cached = self._cache_path(full)
        if use_cache and cached.exists():
            return cached.read_bytes()
        host = urlsplit(full).netloc
        if host in CONTACT_REQUIRED_HOSTS and not self.has_contact:
            raise ContactRequired(
                f"{host} requires contact details in the User-Agent; set FOOTNOTE_FETCH_CONTACT to your "
                "project URL or an email address you read"
            )
        if not await self._allowed(full):
            raise RobotsDisallowed(f"robots.txt disallows {full} for {self.user_agent}")
        lock = self._host_locks.setdefault(host, asyncio.Lock())
        async with lock:
            for attempt in range(5):
                wait = self.min_interval_s - (time.monotonic() - self._last.get(host, 0.0))
                if wait > 0:
                    await asyncio.sleep(wait)
                self._last[host] = time.monotonic()
                try:
                    resp = await self._client.get(full)
                except httpx.TransportError as exc:
                    log.warning("fetch_transport_error", url=full, attempt=attempt, error=str(exc))
                    await asyncio.sleep(2**attempt)
                    continue
                if resp.status_code == 429 or resp.status_code >= 500:
                    retry_after = resp.headers.get("retry-after", "")
                    delay = float(retry_after) if retry_after.isdigit() else 2.0**attempt
                    log.warning("fetch_retry", url=full, status=resp.status_code, delay=delay)
                    await asyncio.sleep(min(delay, 60))
                    continue
                resp.raise_for_status()
                cached.parent.mkdir(parents=True, exist_ok=True)
                cached.write_bytes(resp.content)
                return resp.content
        raise RuntimeError(f"giving up on {full} after 5 attempts")

    async def get_text(self, url: str, params: dict[str, Any] | None = None) -> str:
        return (await self.get_bytes(url, params)).decode("utf-8", errors="replace")

    async def get_json(self, url: str, params: dict[str, Any] | None = None) -> Any:
        return json.loads(await self.get_bytes(url, params))
