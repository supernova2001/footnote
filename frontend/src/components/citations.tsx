"use client";

import { useId, useState, type KeyboardEvent, type ReactNode } from "react";
import { VERDICT_GLYPH, VERDICT_TEXT, numberCitations, verdictTally } from "@/lib/format";
import { usePassage } from "@/lib/use-passage";
import type { Passage, Segment, Verdict, VerdictLabel } from "@/lib/types";
import { LicenseBadge, attributionFor } from "./license-badge";

const VERDICT_STYLE: Record<VerdictLabel, string> = {
  supported: "bg-ok-soft text-ok",
  partial: "bg-warn-soft text-warn",
  unsupported: "bg-bad-soft text-bad",
};

export function VerdictChip({ label }: { label: VerdictLabel }) {
  return (
    <span className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-semibold ${VERDICT_STYLE[label]}`}>
      <span aria-hidden="true">{VERDICT_GLYPH[label]}</span>
      {VERDICT_TEXT[label]}
    </span>
  );
}

function highlight(text: string, quotes: string[]): ReactNode {
  const quote = quotes.map((q) => q.trim()).find((q) => q.length > 0 && text.includes(q));
  if (!quote) return text;
  const at = text.indexOf(quote);
  return (
    <>
      {text.slice(0, at)}
      <mark className="rounded-sm bg-accent-soft px-0.5 text-ink">{quote}</mark>
      {text.slice(at + quote.length)}
    </>
  );
}

export function PassageCard({
  passage,
  number,
  quotes = [],
  compact = false,
}: {
  passage: Passage;
  number?: number;
  quotes?: string[];
  compact?: boolean;
}) {
  return (
    <article className="rounded-md border border-rule bg-surface p-3.5 text-sm">
      <header className="mb-2 flex items-start gap-x-2">
        {number !== undefined && <span className="pt-0.5 font-mono text-xs font-semibold text-accent">{number}.</span>}
        <div className="min-w-0 flex-1">
          <p className="font-serif text-base font-semibold leading-snug">{passage.title}</p>
          {passage.section && <p className="text-xs text-muted">{passage.section}</p>}
        </div>
      </header>
      <blockquote className={`border-l-2 border-rule pl-3 font-serif leading-relaxed ${compact ? "line-clamp-4" : ""}`}>
        {highlight(passage.text, quotes)}
      </blockquote>
      <footer className="mt-2.5 space-y-1 text-xs text-muted">
        <p>{passage.attribution || attributionFor(passage.license_id, passage.title)}</p>
        <p className="flex flex-wrap items-center gap-x-3 gap-y-1">
          <LicenseBadge id={passage.license_id} />
          <a href={passage.source_url} target="_blank" rel="noopener noreferrer" className="text-ink underline decoration-rule hover:decoration-accent">
            Source ↗
          </a>
          <span className="font-mono">{passage.id}</span>
          {passage.score !== null && <span className="font-mono">score {passage.score.toFixed(3)}</span>}
        </p>
      </footer>
    </article>
  );
}

export function VerdictCard({ verdict, claim }: { verdict: Verdict; claim?: string }) {
  return (
    <article className="rounded-md border border-rule bg-surface p-3.5 text-sm">
      <header className="mb-2 flex flex-wrap items-center justify-between gap-2">
        <VerdictChip label={verdict.label} />
        <span className="font-mono text-[11px] text-muted" title="Which model ruled">
          {verdict.judge}
        </span>
      </header>
      {claim && <p className="mb-2 font-serif italic text-muted">“{claim.trim()}”</p>}
      <p>{verdict.rationale}</p>
      {verdict.evidence ? (
        <blockquote className="mt-2 border-l-2 border-ok/50 pl-3 font-serif">“{verdict.evidence}”</blockquote>
      ) : (
        <p className="mt-2 text-xs text-muted">No passage supports this; the judge quoted nothing.</p>
      )}
    </article>
  );
}

type Active = { kind: "passage"; id: string; seg: number } | { kind: "verdict"; seg: number } | null;

function PassageNote({ id, passages, number, quotes }: { id: string; passages: Passage[]; number?: number; quotes: string[] }) {
  const { passage, error } = usePassage(id, passages);
  if (error) return <p className="text-sm text-bad">Couldn’t load passage {id}: {error}</p>;
  if (!passage) return <p className="text-sm text-muted">Loading passage {id}…</p>;
  return <PassageCard passage={passage} number={number} quotes={quotes} />;
}

function Note({ active, segments, passages, numbers }: { active: Active; segments: Segment[]; passages: Passage[]; numbers: Map<string, number> }) {
  if (!active) return null;
  if (active.kind === "verdict") {
    const seg = segments[active.seg];
    if (!seg?.verdict) return null;
    return <VerdictCard verdict={seg.verdict} claim={seg.text} />;
  }
  const quotes = (segments[active.seg]?.citations ?? []).filter((c) => c.passage_id === active.id).map((c) => c.cited_text);
  return <PassageNote id={active.id} passages={passages} number={numbers.get(active.id)} quotes={quotes} />;
}

/**
 * Prose whose sentences carry footnote markers and the judge's ruling.
 * Wide screens show the open note in the margin; narrow screens show it under the text.
 */
export function SegmentedText({
  segments,
  passages,
  showVerdicts = true,
  numbers: numbersIn,
  showNotesList = true,
  pending = false,
}: {
  segments: Segment[];
  passages: Passage[];
  showVerdicts?: boolean;
  numbers?: Map<string, number>;
  showNotesList?: boolean;
  /** True while the judge is still working, so a null verdict means "not judged yet".
   *  Otherwise a null verdict is framing text with no factual claim, and gets no badge. */
  pending?: boolean;
}) {
  const [active, setActive] = useState<Active>(null);
  const uid = useId();
  const marginId = `${uid}-margin`;
  const inlineId = `${uid}-inline`;
  const numbers = numbersIn ?? numberCitations(segments);

  const toggle = (next: Exclude<Active, null>) =>
    setActive((cur) => {
      if (!cur) return next;
      if (cur.kind === next.kind && cur.seg === next.seg && (cur.kind === "verdict" || (next.kind === "passage" && cur.id === next.id))) return null;
      return next;
    });

  const onKeyDown = (e: KeyboardEvent) => {
    if (e.key === "Escape") setActive(null);
  };

  const cited = [...numbers.entries()].sort((a, b) => a[1] - b[1]);
  const note = <Note active={active} segments={segments} passages={passages} numbers={numbers} />;

  // The margin column depends on the width this text gets, not the viewport: a container query.
  return (
    <div onKeyDown={onKeyDown} className="@container">
      <div className="@3xl:grid @3xl:grid-cols-[minmax(0,1fr)_18rem] @3xl:gap-8">
        <p className="prose-serif">
          {segments.map((s, i) => {
            const v = showVerdicts ? s.verdict : null;
            const cls = !showVerdicts
              ? "seg"
              : !s.verdict
                ? pending
                  ? "seg seg-pending"
                  : "seg"
                : s.verdict.label === "supported"
                  ? "seg"
                  : `seg seg-${s.verdict.label}`;
            const isActive = active?.seg === i;
            const ids = [...new Set(s.citations.map((c) => c.passage_id))];
            return (
              <span key={i}>
                <span className={`${cls} ${isActive ? "seg-active" : ""}`}>{s.text.trim()}</span>
                {ids.map((pid) => {
                  const n = numbers.get(pid);
                  const open = active?.kind === "passage" && active.seg === i && active.id === pid;
                  return (
                    <button
                      key={pid}
                      type="button"
                      className="fn-marker"
                      aria-expanded={open}
                      aria-controls={`${marginId} ${inlineId}`}
                      aria-label={`Source ${n ?? ""}: show passage`}
                      onClick={() => toggle({ kind: "passage", id: pid, seg: i })}
                    >
                      {n ?? "?"}
                    </button>
                  );
                })}
                {v && (
                  <button
                    type="button"
                    className={`verdict-glyph ${VERDICT_STYLE[v.label]}`}
                    aria-expanded={active?.kind === "verdict" && active.seg === i}
                    aria-controls={`${marginId} ${inlineId}`}
                    aria-label={`Judge: ${VERDICT_TEXT[v.label]}. Show the ruling`}
                    title={`${VERDICT_TEXT[v.label]}: ${v.rationale}`}
                    onClick={() => toggle({ kind: "verdict", seg: i })}
                  >
                    {VERDICT_GLYPH[v.label]}
                  </button>
                )}
                {showVerdicts && pending && !s.verdict && (
                  <span className="verdict-glyph pulse bg-sunk text-muted" role="img" aria-label="Being judged">
                    …
                  </span>
                )}{" "}
              </span>
            );
          })}
        </p>
        <aside id={marginId} aria-live="polite" className="hidden @3xl:block">
          <div className="sticky top-20">
            {active ? (
              note
            ) : (
              <p className="border-l border-rule pl-3 text-xs leading-relaxed text-muted">
                Select a superscript number to read the passage it cites, or a ✓ ◐ ✕ mark to read the judge’s ruling.
              </p>
            )}
          </div>
        </aside>
      </div>
      <div id={inlineId} aria-live="polite" className="mt-3 @3xl:hidden">
        {note}
      </div>
      {showNotesList && cited.length > 0 && (
        <section className="mt-6 border-t border-rule pt-4" aria-label="Sources cited">
          <h3 className="mb-2 font-mono text-xs uppercase tracking-[0.14em] text-muted">Notes</h3>
          <ol className="space-y-1.5 text-sm">
            {cited.map(([pid, n]) => {
              const p = passages.find((x) => x.id === pid);
              const seg = segments.findIndex((s) => s.citations.some((c) => c.passage_id === pid));
              return (
                <li key={pid} className="flex gap-2">
                  <span className="w-5 shrink-0 text-right font-mono text-xs font-semibold leading-6 text-accent">{n}.</span>
                  <button
                    type="button"
                    className="min-w-0 text-left leading-6 hover:underline"
                    onClick={() => toggle({ kind: "passage", id: pid, seg: Math.max(seg, 0) })}
                  >
                    {p ? (
                      <>
                        <span className="font-serif">{p.title}</span>
                        {p.section && <span className="text-muted">: {p.section}</span>}
                      </>
                    ) : (
                      <span className="font-mono text-xs">{pid}</span>
                    )}
                  </button>
                </li>
              );
            })}
          </ol>
        </section>
      )}
    </div>
  );
}

export function VerdictSummary({ segments, pending = false }: { segments: Segment[]; pending?: boolean }) {
  const t = verdictTally(segments);
  const judged = t.supported + t.partial + t.unsupported;
  const judges = [...new Set(segments.flatMap((s) => (s.verdict ? [s.verdict.judge] : [])))];
  return (
    <p className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-muted">
      <span>
        {judged} claim{judged === 1 ? "" : "s"} checked
      </span>
      {t.supported > 0 && (
        <span className="text-ok">
          {VERDICT_GLYPH.supported} {t.supported} supported
        </span>
      )}
      {t.partial > 0 && (
        <span className="text-warn">
          {VERDICT_GLYPH.partial} {t.partial} partly supported
        </span>
      )}
      {t.unsupported > 0 && (
        <span className="text-bad">
          {VERDICT_GLYPH.unsupported} {t.unsupported} not supported
        </span>
      )}
      {pending && t.pending > 0 && <span>{t.pending} awaiting the judge</span>}
      {judges.length > 0 && <span className="font-mono">judge: {judges.join(", ")}</span>}
    </p>
  );
}
