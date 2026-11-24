"""Fetch and normalise each source into a Document. Each fetcher checks its own licence."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any

from lxml import etree

from footnote.corpus.http import Fetcher
from footnote.corpus.licenses import EUROPE_PMC_ACCEPTED, make_license
from footnote.domain import Document, Section

WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
WIKIDATA_ENTITY = "https://www.wikidata.org/wiki/Special:EntityData/{qid}.json"
GUTENBERG_MIRROR = "https://gutenberg.pglaf.org/cache/epub/{id}/pg{id}.txt"
EUROPE_PMC = "https://www.ebi.ac.uk/europepmc/webservices/rest"


class LicenceError(ValueError):
    """The source does not carry a licence Footnote accepts. The document is skipped, never stored."""


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def slugify(name: str) -> str:
    import unicodedata

    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", ascii_name.lower()).strip("-")


def _assemble(parts: list[tuple[str, str]]) -> tuple[str, list[Section]]:
    """Join (heading, body) parts into one text with section offsets."""
    text, sections = "", []
    for heading, raw_body in parts:
        body = raw_body.strip()
        if not body:
            continue
        start = len(text)
        text += body + "\n\n"
        sections.append(Section(title=heading, start=start, end=len(text)))
    return text, sections


# ── Wikipedia + Wikidata ─────────────────────────────────────────────────────────────────────


@dataclass
class WikipediaArticle:
    title: str
    url: str
    revision_id: int
    qid: str | None
    lead: str
    document: Document


_HEADING = re.compile(r"^(=+)\s*(.+?)\s*\1\s*$", re.M)


async def fetch_wikipedia(fetcher: Fetcher, title: str, scholar_id: str, skip_sections: set[str]) -> WikipediaArticle:
    data = await fetcher.get_json(
        WIKIPEDIA_API,
        {
            "action": "query",
            "prop": "extracts|info|pageprops|revisions",
            "explaintext": 1,
            "exsectionformat": "wiki",
            "inprop": "url",
            "rvprop": "ids|timestamp",
            "redirects": 1,
            "format": "json",
            "formatversion": 2,
            "titles": title,
        },
    )
    page = data["query"]["pages"][0]
    if page.get("missing"):
        raise LookupError(f"Wikipedia has no article {title!r}")
    raw: str = page["extract"]
    parts: list[tuple[str, str]] = []
    heading, pos, skipping = "Introduction", 0, False
    for m in _HEADING.finditer(raw):
        if not skipping:
            parts.append((heading, raw[pos : m.start()]))
        level, heading = len(m.group(1)), m.group(2)
        if level == 2:
            skipping = heading in skip_sections
        pos = m.end()
    if not skipping:
        parts.append((heading, raw[pos:]))
    text, sections = _assemble(parts)
    rev = page["revisions"][0]["revid"]
    url = page["fullurl"]
    doc = Document(
        id=f"wikipedia-{scholar_id}",
        title=f"{page['title']} (Wikipedia)",
        kind="biography",
        year=None,
        scholar_ids=[scholar_id],
        source_url=f"{url}?oldid={rev}",
        license=make_license(
            "CC-BY-SA-4.0",
            f"From the Wikipedia article “{page['title']}” (revision {rev}) by Wikipedia contributors, CC BY-SA 4.0.",
        ),
        sha256=sha256(text),
        text=text,
        sections=sections,
    )
    lead = parts[0][1].strip() if parts else ""
    return WikipediaArticle(
        title=page["title"],
        url=url,
        revision_id=rev,
        qid=page.get("pageprops", {}).get("wikibase_item"),
        lead=lead,
        document=doc,
    )


def _year(claims: dict[str, Any], prop: str) -> int | None:
    for claim in claims.get(prop, []):
        value = claim.get("mainsnak", {}).get("datavalue", {}).get("value", {})
        m = re.match(r"^[+-](\d{1,4})-", value.get("time", ""))
        if m:
            return int(m.group(1))
    return None


async def fetch_life_years(fetcher: Fetcher, qid: str | None) -> tuple[int | None, int | None]:
    """Birth and death years from Wikidata (CC0)."""
    if not qid:
        return None, None
    data = await fetcher.get_json(WIKIDATA_ENTITY.format(qid=qid))
    claims = data["entities"][qid]["claims"]
    return _year(claims, "P569"), _year(claims, "P570")


# ── Project Gutenberg (PGLAF mirror) ─────────────────────────────────────────────────────────

_PD_NOTICE = "This eBook is for the use of anyone anywhere in the United States"
_START = re.compile(r"^\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG EBOOK.*$", re.M | re.I)
_END = re.compile(r"^\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG EBOOK.*$", re.M | re.I)
_CREDIT = re.compile(
    r"^(Produced by|E-text prepared by|Transcriber's note|\[Illustration).*?(\n\s*\n|\Z)", re.I | re.S | re.M
)
# Older e-texts repeat the end marker in prose ("End of Project Gutenberg's Southern Horrors, by ...").
_PG_FOOTER = re.compile(r"^\s*End of (the |this )?Project Gutenberg.*$", re.I | re.M)
# Notes the volunteer transcriber or editor added in brackets: about the e-text, not the author's words.
_TRANSCRIBER = re.compile(r"\[(Transcriber|Editor)['’]?s? notes?:?.*?\]", re.I | re.S)


_MARKUP = re.compile(r"[_+=]([^_+=\n]+)[_+=]")  # Gutenberg's _italic_, +small caps+, =bold=
_NAV_SECTIONS = frozenset(
    {
        "contents",
        "table of contents",
        "illustrations",
        "list of illustrations",
        "index",
        "maps and diagrams",
        "chapter page",
        "maps",
        "list of maps",
        "errata",
    }
)


def _heading_text(block: str) -> str:
    """'+INTRODUCTION+' / '4 _THE_ MALICIOUS...' / "DOUGLASS'S LETTER" → readable title case."""
    import string

    text = _MARKUP.sub(r"\1", " ".join(block.split())).strip(" _+=")
    text = re.sub(r"^\d+\s+", "", text)  # a leading page or section number
    return string.capwords(text.lower())


def _is_heading(block: str) -> bool:
    """A Gutenberg heading: one or two short lines, mostly capitals (CHAPTER IV / OF OUR SPIRITUAL STRIVINGS)."""
    lines = [ln.strip() for ln in block.strip().splitlines()]
    if not 1 <= len(lines) <= 3 or any(len(ln) > 80 for ln in lines):
        return False
    letters = [ch for ch in " ".join(lines) if ch.isalpha()]
    return bool(letters) and sum(ch.isupper() for ch in letters) / len(letters) > 0.8


async def fetch_gutenberg(
    fetcher: Fetcher, ebook_id: int, *, title: str, year: int | None, scholar_id: str, author: str | None = None
) -> Document:
    raw = await fetcher.get_text(GUTENBERG_MIRROR.format(id=ebook_id))
    raw = raw.replace("\r\n", "\n")
    header = raw[:4000]
    if _PD_NOTICE not in " ".join(header.split()):
        raise LicenceError(f"Gutenberg #{ebook_id} does not carry the US public-domain notice")
    start, end = _START.search(raw), _END.search(raw)
    if not start or not end:
        raise ValueError(f"Gutenberg #{ebook_id}: START/END markers not found")
    body = _CREDIT.sub("", _TRANSCRIBER.sub("", _PG_FOOTER.sub("", raw[start.end() : end.start()]))).strip()
    # Unwrap hard-wrapped lines inside paragraphs; keep blank lines as paragraph breaks.
    blocks = [b for b in re.split(r"\n\s*\n", body) if b.strip()]
    parts: list[tuple[str, str]] = []
    heading, buf = "Front matter", list[str]()
    for block in blocks:
        if _is_heading(block):
            if buf:
                parts.append((heading, "\n\n".join(buf)))
                buf = []
            heading = _heading_text(block)
        else:
            buf.append(_MARKUP.sub(r"\1", " ".join(block.split())))
    if buf:
        parts.append((heading, "\n\n".join(buf)))
    # Navigation pages (contents, index, lists of maps) are lists of page numbers, not prose:
    # left in, they win keyword searches for every name they list.
    parts = [(h, b) for h, b in parts if h.lower().strip(" .:") not in _NAV_SECTIONS]
    # A title page (title, author, year, publisher in a few lines) is metadata the chunks already carry.
    parts = [(h, b) for i, (h, b) in enumerate(parts) if not (i == 0 and h == "Front matter" and len(b.split()) < 60)]
    text, sections = _assemble(parts)
    return Document(
        id=f"gutenberg-{ebook_id}",
        title=title,
        kind="work",
        year=year,
        scholar_ids=[scholar_id],
        source_url=f"https://www.gutenberg.org/ebooks/{ebook_id}",
        license=make_license(
            "PD-US",
            f"{author + ', ' if author else ''}“{title}” ({year}), public domain in the United States; "
            f"text from Project Gutenberg eBook #{ebook_id}.",
        ),
        sha256=sha256(text),
        text=text,
        sections=sections,
    )


# ── Europe PMC (open-access papers) ──────────────────────────────────────────────────────────

_BACK_MATTER = re.compile(
    r"^(acknowledg|author contribution|funding|conflicts? of interest|competing interest|data availability|"
    r"institutional review|informed consent|ethics|supplementary|abbreviations|references)",
    re.I,
)


@dataclass
class PaperMeta:
    pmcid: str
    title: str
    year: int | None
    authors: list[str]
    licence: str
    journal: str | None
    doi: str | None


async def search_papers(fetcher: Fetcher, query: str, sort: str, limit: int) -> list[PaperMeta]:
    data = await fetcher.get_json(
        f"{EUROPE_PMC}/search",
        {"query": query, "format": "json", "resultType": "core", "pageSize": min(100, limit * 4), "sort": sort},
    )
    out: list[PaperMeta] = []
    for r in data["resultList"]["result"]:
        title = (r.get("title") or "").strip().rstrip(".")
        if not r.get("pmcid") or re.match(r"^(correction|erratum|retraction|expression of concern)", title, re.I):
            continue
        if (r.get("license") or "").lower() not in EUROPE_PMC_ACCEPTED:
            continue
        authors = [a["fullName"] for a in r.get("authorList", {}).get("author", []) if a.get("fullName")]
        out.append(
            PaperMeta(
                pmcid=r["pmcid"],
                title=title,
                year=int(r["pubYear"]) if r.get("pubYear") else None,
                authors=authors,
                licence=r["license"].lower(),
                journal=r.get("journalInfo", {}).get("journal", {}).get("title"),
                doi=r.get("doi"),
            )
        )
        if len(out) >= limit:
            break
    return out


def _text(el: etree._Element) -> str:
    return " ".join(" ".join(el.itertext()).split())


def _xml_licence_ok(root: etree._Element) -> bool:
    hrefs = [h for lic in root.iter("license") for h in lic.xpath(".//@*[local-name()='href']")]
    hrefs += [_text(r) for r in root.iter("{http://www.niso.org/schemas/ali/1.0/}license_ref")]
    for raw in hrefs:
        href = raw.lower()
        if "-nc" in href or "-nd" in href:
            return False
        if "/licenses/by/" in href or "publicdomain/zero" in href:
            return True
    return False


async def fetch_paper(fetcher: Fetcher, meta: PaperMeta, author_ids: list[str]) -> Document:
    xml = await fetcher.get_bytes(f"{EUROPE_PMC}/{meta.pmcid}/fullTextXML")
    root = etree.fromstring(xml)
    if not _xml_licence_ok(root):
        raise LicenceError(f"{meta.pmcid}: the article XML does not carry a CC BY / CC0 licence")
    parts: list[tuple[str, str]] = []
    abstract = root.find(".//front//abstract")
    if abstract is not None:
        parts.append(("Abstract", "\n\n".join(_text(p) for p in abstract.iter("p"))))
    body = root.find(".//body")
    if body is not None:
        for sec in body.findall("sec"):
            title_el = sec.find("title")
            heading = _text(title_el) if title_el is not None else ""
            if _BACK_MATTER.match(heading):
                continue
            paras = [_text(p) for p in sec.iter("p")]
            parts.append((heading or "Body", "\n\n".join(p for p in paras if p)))
    text, sections = _assemble(parts)
    license_id = EUROPE_PMC_ACCEPTED[meta.licence]
    byline = ", ".join(meta.authors[:3]) + (" et al." if len(meta.authors) > 3 else "")
    return Document(
        id=f"pmc-{meta.pmcid}",
        title=meta.title,
        kind="paper",
        year=meta.year,
        scholar_ids=author_ids,
        source_url=f"https://europepmc.org/article/PMC/{meta.pmcid}",
        license=make_license(
            license_id,
            f"{byline}, “{meta.title}”{', ' + meta.journal if meta.journal else ''} ({meta.year}). "
            f"{'CC BY 4.0' if license_id == 'CC-BY-4.0' else 'CC0 1.0'}. "
            f"{'doi:' + meta.doi if meta.doi else meta.pmcid}",
        ),
        sha256=sha256(text),
        text=text,
        sections=sections,
    )
