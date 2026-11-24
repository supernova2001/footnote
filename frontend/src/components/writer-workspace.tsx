"use client";

import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { useEffect, useId, useLayoutEffect, useMemo, useRef, useState, type KeyboardEvent, type ReactNode } from "react";
import { api } from "@/lib/api";
import { AUDIENCE_LABEL, VERDICT_GLYPH, numberCitations, shortDate } from "@/lib/format";
import type { Audience, Draft, Passage, Segment, SectionStatus } from "@/lib/types";
import { useApi } from "@/lib/use-api";
import { useDraftStream, type DraftStreamState, type ThreadEntry } from "@/lib/use-draft";
import { PassageCard, SegmentedText, VerdictCard, VerdictSummary } from "./citations";

/**
 * The writer's workspace: the story in the middle, the writer agent in a pane beside it.
 *
 * You set the story up and steer it by talking to the writer (Chat). The plan and then each section
 * land on the canvas where the finished story will be, so you watch it being written and checked
 * in place. After drafting, the outline on the left shows each section's state, and the pane's
 * Checks and Sources tabs show what the judge flagged and what the story may cite. The graph pauses
 * for review; Approve & publish (top bar) is the one way to publish, and asking for changes in Chat
 * is the one way to revise.
 */

const AUDIENCES: Audience[] = ["grade4", "grade8", "adult"];
const WRITER = "Footnote writer";

const STARTERS = [
  "How Ida B. Wells used newspapers to investigate injustice, for a class of thirteen-year-olds.",
  "Carter G. Woodson and why he started Negro History Week, for curious adults.",
  "What schooling looked like for Black children before 1861, for grade 4 readers.",
  "Why Ida B. Wells wrote The Red Record, explained for grade 8.",
];
const REVISIONS = [
  "Make it shorter and simpler.",
  "Explain any hard words the first time they appear.",
  "Open with a moment from her life before the argument.",
  "Add more about why this still matters today.",
];

type Tab = "chat" | "ideas" | "checks" | "sources";
type ViewSection = {
  key: string;
  heading: string;
  goal: string | null;
  segments: Segment[];
  warnings: string[];
  status: SectionStatus | "pending" | "final";
  attempt: number;
};

// ── small pieces ──────────────────────────────────────────────────────────────────────────

const ICON = "h-4 w-4 shrink-0";
const icons: Record<Tab, ReactNode> = {
  chat: (
    <svg viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.6" className={ICON} aria-hidden="true">
      <path d="M4 5.5A2.5 2.5 0 0 1 6.5 3h7A2.5 2.5 0 0 1 16 5.5v5a2.5 2.5 0 0 1-2.5 2.5H9l-3.5 3v-3A2.5 2.5 0 0 1 4 10.5z" strokeLinejoin="round" />
    </svg>
  ),
  ideas: (
    <svg viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.6" className={ICON} aria-hidden="true">
      <path d="M7.5 14.5h5M8.5 17h3M10 3a5 5 0 0 0-3 9c.6.5 1 1.1 1 1.8v.2h4v-.2c0-.7.4-1.3 1-1.8a5 5 0 0 0-3-9z" strokeLinejoin="round" />
    </svg>
  ),
  checks: (
    <svg viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.6" className={ICON} aria-hidden="true">
      <circle cx="10" cy="10" r="6.5" />
      <path d="m7.3 10.2 1.9 1.9 3.6-3.8" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  ),
  sources: (
    <svg viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.6" className={ICON} aria-hidden="true">
      <path d="M10 5.5C8.5 4.3 6.3 4 4 4.3v10.5c2.3-.3 4.5 0 6 1.2 1.5-1.2 3.7-1.5 6-1.2V4.3c-2.3-.3-4.5 0-6 1.2zM10 5.5V16" strokeLinejoin="round" />
    </svg>
  ),
};

/** The writer's mark: Footnote's superscript one, with a ring while it works. */
function WriterMark({ busy = false, size = "md" }: { busy?: boolean; size?: "sm" | "md" }) {
  const box = size === "sm" ? "h-6 w-6 text-xs" : "h-10 w-10 text-lg";
  return (
    <span aria-hidden="true" className={`relative inline-flex ${box} shrink-0 items-center justify-center rounded-full border border-rule bg-paper font-serif font-semibold text-accent`}>
      ¹{busy && <span className="pulse absolute -inset-0.5 rounded-full border-2 border-accent/50" />}
    </span>
  );
}

function Dot({ tone }: { tone: "ok" | "warn" | "bad" | "busy" | "faint" }) {
  const cls = {
    ok: "bg-ok",
    warn: "bg-warn",
    bad: "bg-bad",
    busy: "pulse border-2 border-accent bg-paper",
    faint: "bg-rule",
  }[tone];
  return <span aria-hidden="true" className={`inline-block h-2 w-2 shrink-0 rounded-full ${cls}`} />;
}

function sectionTone(s: ViewSection): "ok" | "warn" | "bad" | "busy" | "faint" {
  if (s.status === "pending") return "faint";
  if (s.status === "drafting" || s.status === "judging" || s.status === "redrafting") return "busy";
  if (s.status === "dropped") return "bad";
  const flagged = s.warnings.length > 0 || s.segments.some((g) => g.verdict && g.verdict.label !== "supported");
  return flagged ? "warn" : "ok";
}

const words = (segments: Segment[]) => segments.reduce((n, s) => n + (s.text.trim().match(/\S+/g)?.length ?? 0), 0);

// ── what the page shows, derived from the stream ──────────────────────────────────────────

function viewSections(state: DraftStreamState): ViewSection[] {
  if (state.draft && (state.status === "review" || state.status === "published" || !state.outline)) {
    return state.draft.sections.map((s, i) => ({
      key: `final-${i}`,
      heading: s.heading,
      goal: null,
      segments: s.segments,
      warnings: s.warnings,
      status: "final",
      attempt: 1,
    }));
  }
  if (!state.outline) return [];
  return state.outline.sections.map((o, i) => {
    const live = state.sections[i];
    return {
      key: `live-${i}`,
      heading: live?.heading ?? o.heading,
      goal: o.goal,
      segments: live?.segments ?? [],
      warnings: [],
      status: live?.status ?? "pending",
      attempt: live?.attempt ?? 0,
    };
  });
}

function agentStatus(state: DraftStreamState): { text: string; tone: "busy" | "warn" | "ok" | "bad" | null } {
  if (state.status === "idle") return { text: "Ready when you are", tone: null };
  if (state.status === "review") return { text: "Waiting for your review", tone: "warn" };
  if (state.status === "published") return { text: "Published", tone: "ok" };
  if (state.status === "error") return { text: state.error ?? "Stopped", tone: "bad" };
  const running = [...state.steps].reverse().find((s) => s.status === "start");
  const total = state.outline?.sections.length ?? 0;
  const busy = state.sections.filter((s) => s.status === "drafting" || s.status === "judging" || s.status === "redrafting");
  if (busy.length > 1) return { text: `Writing ${busy.length} sections in parallel`, tone: "busy" };
  const active = busy[0];
  if (active) {
    const n = `section ${active.index + 1} of ${total}`;
    if (active.status === "judging") return { text: `Checking ${n} against its sources`, tone: "busy" };
    if (active.status === "redrafting") return { text: `Rewriting ${n}: a claim failed`, tone: "busy" };
    return { text: `Drafting ${n}`, tone: "busy" };
  }
  const byNode: Record<string, string> = {
    research: "Researching the collection",
    outline: "Planning the story",
    write_section: "Drafting",
    assemble: "Checking the reading level",
    review: "Saving the draft for you",
    publish: "Publishing",
  };
  return { text: (running && byNode[running.node]) || "Working", tone: "busy" };
}

type Check = { key: string; section: number; heading: string; text: string; segment?: Segment; warning?: string };

function checksFor(sections: ViewSection[]): Check[] {
  const out: Check[] = [];
  sections.forEach((s, i) => {
    s.warnings.forEach((w, j) => out.push({ key: `w-${i}-${j}`, section: i, heading: s.heading, text: w, warning: w }));
    s.segments.forEach((g, j) => {
      if (g.verdict && g.verdict.label !== "supported") out.push({ key: `s-${i}-${j}`, section: i, heading: s.heading, text: g.text.trim(), segment: g });
    });
    if (s.status === "dropped") out.push({ key: `d-${i}`, section: i, heading: s.heading, text: "Nothing in this section could be supported, so it was dropped." });
  });
  return out;
}

// ── the canvas: the story, written in place ───────────────────────────────────────────────

function SectionBody({ s, passages, numbers }: { s: ViewSection; passages: Passage[]; numbers: Map<string, number> }) {
  const busy = s.status === "drafting" || s.status === "judging" || s.status === "redrafting";
  if (s.status === "pending") {
    return <p className="font-serif text-lg italic leading-relaxed text-faint">{s.goal}</p>;
  }
  if (s.status === "dropped") {
    return <p className="text-sm text-bad">Nothing in this section could be supported by its sources, so it was dropped.</p>;
  }
  return (
    <div>
      {busy && (
        <p className={`mb-2 flex items-center gap-2 text-xs font-medium ${s.status === "redrafting" ? "text-warn" : "text-accent"}`} role="status">
          <Dot tone="busy" />
          {s.status === "drafting" && (s.attempt > 1 ? `Rewriting (attempt ${s.attempt})…` : "Writing…")}
          {s.status === "judging" && "Checking each claim against its source…"}
          {s.status === "redrafting" && "A claim didn’t hold up, so this section is being rewritten with it named…"}
        </p>
      )}
      {s.segments.length > 0 ? (
        <div className={s.status === "redrafting" ? "opacity-50" : ""}>
          <SegmentedText segments={s.segments} passages={passages} numbers={numbers} showNotesList={false} pending={busy} />
        </div>
      ) : (
        <div className="space-y-2.5" aria-hidden="true">
          {[92, 100, 84, 60].map((w, i) => (
            <div key={i} className="pulse h-3.5 rounded bg-sunk" style={{ width: `${w}%` }} />
          ))}
        </div>
      )}
      {s.warnings.map((w) => (
        <p key={w} className="mt-2 rounded-lg border border-warn/30 bg-warn-soft px-3 py-1.5 text-xs">
          {w}
        </p>
      ))}
    </div>
  );
}

function Canvas({ state, sections, audience, onIdea }: { state: DraftStreamState; sections: ViewSection[]; audience: Audience; onIdea: (t: string) => void }) {
  const title = state.draft?.title ?? state.outline?.title ?? null;
  const passages = state.draft?.sources ?? state.sources;
  const numbers = numberCitations(sections.flatMap((s) => s.segments));
  const researching = state.status === "streaming" && !state.outline;

  if (state.status === "idle" || (!title && !researching)) {
    return (
      <div className="mx-auto flex h-full max-w-xl flex-col items-center justify-center px-6 py-12 text-center">
        <p className="font-serif text-3xl font-semibold tracking-tight text-faint">A blank page, for now.</p>
        <p className="mt-3 text-sm leading-relaxed text-muted">
          Tell the writer what the story should cover and who it is for. It researches the collection, plans the story
          here, and drafts each section in place, where you can watch every claim being checked.
        </p>
        <ul className="mt-6 grid w-full gap-2 text-left">
          {STARTERS.slice(0, 3).map((s) => (
            <li key={s}>
              <button type="button" onClick={() => onIdea(s)} className="w-full rounded-xl border border-rule bg-surface px-4 py-2.5 text-sm leading-snug transition hover:border-accent/40 hover:bg-sunk">
                {s}
              </button>
            </li>
          ))}
        </ul>
      </div>
    );
  }

  return (
    <article className="mx-auto max-w-2xl px-6 pb-24 pt-10">
      <p className="font-mono text-xs uppercase tracking-[0.14em] text-muted">
        {AUDIENCE_LABEL[state.draft?.audience ?? audience] ?? audience}
        {sections.length > 0 && ` · ${sections.length} sections`}
      </p>
      {researching ? (
        <div className="mt-3 space-y-3" aria-hidden="true">
          <div className="pulse h-9 w-3/4 rounded-lg bg-sunk" />
          <p className="text-sm text-muted">Researching the collection for this story…</p>
        </div>
      ) : (
        <h1 className="mt-2 font-serif text-3xl font-semibold leading-tight tracking-tight sm:text-4xl">{title}</h1>
      )}
      {state.draft?.brief && <p className="mt-3 text-sm text-muted">{state.draft.brief}</p>}
      <div className="mt-8 space-y-9">
        {sections.map((s, i) => (
          <section key={s.key} id={`story-sec-${i}`} className="scroll-mt-6">
            <h2 className={`mb-3 font-serif text-xl font-semibold ${s.status === "pending" ? "text-muted" : ""}`}>{s.heading}</h2>
            <SectionBody s={s} passages={passages} numbers={numbers} />
          </section>
        ))}
      </div>
    </article>
  );
}

// ── the writer's pane ─────────────────────────────────────────────────────────────────────

function ThreadLine({ entry }: { entry: ThreadEntry }) {
  switch (entry.kind) {
    case "user":
      return (
        <div className="flex justify-end">
          <p className="max-w-[88%] whitespace-pre-wrap rounded-2xl rounded-br-md bg-sunk px-3.5 py-2 text-sm leading-relaxed">{entry.text}</p>
        </div>
      );
    case "sources":
      return (
        <p className="text-sm leading-relaxed">
          I found <b>{entry.count} passages</b> across <b>{entry.works} work{entry.works === 1 ? "" : "s"}</b> in the collection. The story can cite only these.
        </p>
      );
    case "outline":
      return (
        <div className="text-sm leading-relaxed">
          <p>
            Here’s the plan for <b className="font-serif">{entry.title}</b>:
          </p>
          <ol className="mt-1.5 list-decimal space-y-0.5 pl-5 text-ink">
            {entry.headings.map((h) => (
              <li key={h}>{h}</li>
            ))}
          </ol>
          <p className="mt-1.5 text-muted">I’m drafting each section now. Every claim is checked before it stays.</p>
        </div>
      );
    case "section": {
      const tone = entry.status === "accepted" ? "text-ok" : entry.status === "redrafting" ? "text-warn" : "text-bad";
      const glyph = entry.status === "accepted" ? VERDICT_GLYPH.supported : entry.status === "redrafting" ? VERDICT_GLYPH.partial : VERDICT_GLYPH.unsupported;
      const text =
        entry.status === "accepted"
          ? `“${entry.heading}” passed the check.`
          : entry.status === "redrafting"
            ? `A claim in “${entry.heading}” didn’t hold up against its source, so I’m rewriting it with that claim named.`
            : `Nothing in “${entry.heading}” could be supported, so I dropped it.`;
      return (
        <p className="flex gap-2 text-sm leading-relaxed">
          <span className={`${tone} shrink-0 font-semibold`} aria-hidden="true">
            {glyph}
          </span>
          <span className={entry.status === "accepted" ? "text-muted" : ""}>{text}</span>
        </p>
      );
    }
    case "readability":
      return (
        <p className="text-sm leading-relaxed">
          {entry.fit === "fits"
            ? "Reading level checked: "
            : entry.retrying
              ? "The reading level is off, so I’m doing one more round: "
              : "The reading level is still off, and it’s flagged under Checks: "}
          grade {entry.grade.toFixed(1)} for {AUDIENCE_LABEL[entry.target] ?? entry.target} readers. <span className="text-muted">{entry.note}</span>
        </p>
      );
    case "review":
      return (
        <p className="text-sm leading-relaxed">
          The draft is ready. Read it in the story view, then use <b>Approve &amp; publish</b> at the top, or tell me here what to
          change.
        </p>
      );
    case "published":
      return <p className="text-sm leading-relaxed text-ok">Published. Start a new story whenever you like.</p>;
    case "error":
      return (
        <p className="text-sm leading-relaxed text-bad" role="alert">
          {entry.message}
        </p>
      );
  }
}

function Thread({ state }: { state: DraftStreamState }) {
  const status = agentStatus(state);
  const groups: { assistant: boolean; entries: ThreadEntry[] }[] = [];
  for (const e of state.thread) {
    const assistant = e.kind !== "user";
    const last = groups.at(-1);
    if (last && last.assistant === assistant) last.entries.push(e);
    else groups.push({ assistant, entries: [e] });
  }
  return (
    <div className="space-y-4">
      <div className="flex gap-2.5">
        <WriterMark size="sm" />
        <p className="text-sm leading-relaxed">
          What should we write? Tell me the topic and who it’s for. I’ll research the collection, plan the story, draft each
          section, and have every claim checked against its source before you see it.
        </p>
      </div>
      {groups.map((g, i) =>
        g.assistant ? (
          <div key={i} className="flex gap-2.5">
            <WriterMark size="sm" />
            <div className="min-w-0 flex-1 space-y-2.5">
              {g.entries.map((e, j) => (
                <ThreadLine key={j} entry={e} />
              ))}
            </div>
          </div>
        ) : (
          <div key={i} className="space-y-2">
            {g.entries.map((e, j) => (
              <ThreadLine key={j} entry={e} />
            ))}
          </div>
        ),
      )}
      {state.status === "streaming" && (
        <div className="flex items-center gap-2.5 pl-0.5 text-sm text-muted" role="status">
          <span className="inline-flex gap-1" aria-hidden="true">
            {[0, 1, 2].map((i) => (
              <span key={i} className="pulse h-1.5 w-1.5 rounded-full bg-accent" style={{ animationDelay: `${i * 160}ms` }} />
            ))}
          </span>
          {status.text}…
        </div>
      )}
    </div>
  );
}

function Setup({
  audience,
  setAudience,
  scholarId,
  setScholarId,
  documentId,
  setDocumentId,
}: {
  audience: Audience;
  setAudience: (a: Audience) => void;
  scholarId: string;
  setScholarId: (s: string) => void;
  documentId: string;
  setDocumentId: (d: string) => void;
}) {
  const uid = useId();
  const scholars = useApi("scholars", (s) => api.scholars(s));
  const detail = useApi(`scholar:${scholarId}`, (s) => (scholarId ? api.scholar(scholarId, s) : Promise.resolve(null)));
  const docs = detail.status === "ok" && detail.data ? detail.data.documents : [];
  const select = "min-w-0 flex-1 rounded-lg border border-rule bg-paper px-2 py-1 text-xs text-ink focus:border-focus focus:outline-none";
  return (
    <div className="space-y-2 px-1 pb-1">
      <div className="flex items-center gap-2" role="radiogroup" aria-label="Who the story is for">
        <span className="w-9 shrink-0 text-xs text-muted">For</span>
        {AUDIENCES.map((a) => (
          <button
            key={a}
            type="button"
            role="radio"
            aria-checked={audience === a}
            onClick={() => setAudience(a)}
            className={`rounded-full border px-2.5 py-0.5 text-xs transition ${audience === a ? "border-ink bg-ink text-paper" : "border-rule text-muted hover:text-ink"}`}
          >
            {AUDIENCE_LABEL[a]}
          </button>
        ))}
      </div>
      <div className="flex items-center gap-2">
        <label htmlFor={`${uid}-s`} className="w-9 shrink-0 text-xs text-muted">
          From
        </label>
        <select
          id={`${uid}-s`}
          className={select}
          value={scholarId}
          disabled={scholars.status !== "ok"}
          onChange={(e) => {
            setScholarId(e.target.value);
            setDocumentId("");
          }}
        >
          <option value="">Any scholar</option>
          {scholars.status === "ok" &&
            scholars.data.map((s) => (
              <option key={s.id} value={s.id}>
                {s.name}
              </option>
            ))}
        </select>
        <label htmlFor={`${uid}-d`} className="sr-only">
          Source document
        </label>
        <select id={`${uid}-d`} className={select} value={documentId} disabled={!scholarId} onChange={(e) => setDocumentId(e.target.value)}>
          <option value="">{scholarId ? "Any of their works" : "Any work"}</option>
          {docs.map((d) => (
            <option key={d.id} value={d.id}>
              {d.title}
            </option>
          ))}
          {documentId && !docs.some((d) => d.id === documentId) && <option value={documentId}>{documentId}</option>}
        </select>
      </div>
    </div>
  );
}

function Composer({
  mode,
  value,
  setValue,
  onSend,
  onStop,
  children,
}: {
  mode: "brief" | "revise" | "busy";
  value: string;
  setValue: (v: string) => void;
  onSend: () => void;
  onStop: () => void;
  children?: ReactNode;
}) {
  const ref = useRef<HTMLTextAreaElement>(null);
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 160)}px`;
  }, [value]);
  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      if (mode !== "busy" && value.trim()) onSend();
    }
  };
  const placeholder =
    mode === "brief" ? "What should the story cover, and for whom?" : mode === "revise" ? "Tell the writer what to change…" : "The writer is working…";
  return (
    <div className="border-t border-rule p-3">
      <div className="rounded-2xl border border-rule bg-paper px-3 pb-2 pt-2.5 transition focus-within:border-focus focus-within:ring-2 focus-within:ring-focus/15">
        <label className="sr-only" htmlFor="writer-input">
          {placeholder}
        </label>
        <textarea
          id="writer-input"
          ref={ref}
          rows={1}
          value={value}
          maxLength={1000}
          disabled={mode === "busy"}
          placeholder={placeholder}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={onKeyDown}
          className="chat-input block max-h-40 w-full resize-none bg-transparent text-sm leading-relaxed text-ink placeholder:text-faint focus:outline-none disabled:cursor-not-allowed"
        />
        <div className="mt-1.5 flex items-end justify-between gap-2">
          <div className="min-w-0 flex-1">{children}</div>
          {mode === "busy" ? (
            <button type="button" onClick={onStop} aria-label="Stop" className="inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-ink text-paper hover:opacity-85">
              <svg viewBox="0 0 20 20" fill="currentColor" className="h-3.5 w-3.5" aria-hidden="true">
                <rect x="4" y="4" width="12" height="12" rx="2" />
              </svg>
            </button>
          ) : (
            <button
              type="button"
              onClick={onSend}
              disabled={!value.trim()}
              aria-label={mode === "brief" ? "Start writing" : "Send changes"}
              className="inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-ink text-paper hover:opacity-85 disabled:cursor-not-allowed disabled:opacity-30"
            >
              <svg viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="2" className="h-4 w-4" aria-hidden="true">
                <path d="M10 16V4M4.5 9.5 10 4l5.5 5.5" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

function ChecksTab({ sections, state, onGo }: { sections: ViewSection[]; state: DraftStreamState; onGo: (i: number) => void }) {
  const checks = checksFor(sections);
  const segments = sections.flatMap((s) => s.segments);
  return (
    <div className="space-y-4">
      {segments.length > 0 && <VerdictSummary segments={segments} pending={state.status === "streaming"} />}
      {state.readability && (
        <div className="rounded-xl border border-rule bg-paper p-3 text-sm">
          <p className="font-medium">
            Reading level: grade {state.readability.grade_level.toFixed(1)}
            <span className="font-normal text-muted"> for {AUDIENCE_LABEL[state.readability.target] ?? state.readability.target}</span>
          </p>
          <p className="mt-1 text-muted">{state.readability.editor_note}</p>
        </div>
      )}
      {checks.length === 0 ? (
        <p className="text-sm text-muted">
          {state.status === "streaming"
            ? "Checks appear here as each section is judged."
            : segments.length
              ? "Nothing needs a look: every claim is supported by its source."
              : "Nothing to check yet."}
        </p>
      ) : (
        <ul className="space-y-2.5">
          {checks.map((c) => (
            <li key={c.key} className="rounded-xl border border-rule bg-paper p-3 text-sm">
              <button type="button" onClick={() => onGo(c.section)} className="mb-1 text-xs text-muted hover:text-ink hover:underline">
                {c.heading} →
              </button>
              {c.segment?.verdict ? (
                <>
                  <p className="mb-2 font-serif leading-relaxed">“{c.text}”</p>
                  <VerdictCard verdict={c.segment.verdict} />
                </>
              ) : (
                <p className={c.warning ? "text-warn" : "text-bad"}>{c.text}</p>
              )}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function SourcesTab({ passages, numbers }: { passages: Passage[]; numbers: Map<string, number> }) {
  if (passages.length === 0) return <p className="text-sm text-muted">The research agent’s picks appear here once it has searched.</p>;
  const cited = passages.filter((p) => numbers.has(p.id)).sort((a, b) => numbers.get(a.id)! - numbers.get(b.id)!);
  const other = passages.filter((p) => !numbers.has(p.id));
  return (
    <div className="space-y-3">
      {cited.length > 0 && <p className="text-xs text-muted">Cited in the story</p>}
      <ol className="space-y-2.5">
        {cited.map((p) => (
          <li key={p.id}>
            <PassageCard passage={p} number={numbers.get(p.id)} compact />
          </li>
        ))}
      </ol>
      {other.length > 0 && (
        <details open={cited.length === 0} className="rounded-xl border border-rule">
          <summary className="cursor-pointer px-3 py-2 text-xs text-muted hover:text-ink">
            {cited.length ? `Also found, not cited (${other.length})` : `Found by research (${other.length})`}
          </summary>
          <ul className="space-y-2.5 p-3 pt-0">
            {other.map((p) => (
              <li key={p.id}>
                <PassageCard passage={p} compact />
              </li>
            ))}
          </ul>
        </details>
      )}
    </div>
  );
}

// ── the top bar ───────────────────────────────────────────────────────────────────────────

function DraftsMenu({ onNew }: { onNew: () => void }) {
  const [open, setOpen] = useState(false);
  const drafts = useApi(open ? "drafts:open" : "drafts:closed", (s) => (open ? api.drafts(s) : Promise.resolve([])));
  return (
    <div className="relative">
      <button
        type="button"
        aria-haspopup="menu"
        aria-expanded={open}
        onClick={() => setOpen((v) => !v)}
        className="inline-flex items-center gap-1 rounded-lg px-2 py-1.5 text-sm text-muted transition hover:bg-sunk hover:text-ink"
      >
        Drafts
        <svg width="10" height="10" viewBox="0 0 10 10" aria-hidden="true">
          <path d="M2 3.5 5 6.5 8 3.5" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </button>
      {open && (
        <div role="menu" className="absolute left-0 top-full z-20 mt-1 w-72 rounded-xl border border-rule bg-surface p-1.5 shadow-lg">
          <button
            type="button"
            role="menuitem"
            onClick={() => {
              setOpen(false);
              onNew();
            }}
            className="block w-full rounded-lg px-3 py-2 text-left text-sm font-medium hover:bg-sunk"
          >
            + New story
          </button>
          {drafts.status === "ok" && drafts.data.length > 0 && <hr className="my-1 border-rule" />}
          {drafts.status === "loading" && <p className="px-3 py-2 text-xs text-muted">Loading…</p>}
          {drafts.status === "ok" &&
            drafts.data.slice(0, 8).map((d) => (
              <Link key={d.id} role="menuitem" href={`/write/${encodeURIComponent(d.id)}`} onClick={() => setOpen(false)} className="block rounded-lg px-3 py-2 hover:bg-sunk">
                <span className="block truncate text-sm">{d.title ?? "Untitled draft"}</span>
                <span className="text-xs text-muted">
                  {d.status.replace("_", " ")} · {shortDate(d.updated_at)}
                </span>
              </Link>
            ))}
        </div>
      )}
    </div>
  );
}

const PILL: Record<DraftStreamState["status"], { text: string; cls: string }> = {
  idle: { text: "New story", cls: "border-rule text-muted" },
  streaming: { text: "Writing", cls: "border-accent/40 bg-accent-soft text-accent" },
  review: { text: "Waiting for review", cls: "border-warn/40 bg-warn-soft text-warn" },
  published: { text: "Published", cls: "border-ok/40 bg-ok-soft text-ok" },
  error: { text: "Stopped", cls: "border-bad/40 bg-bad-soft text-bad" },
};

// ── the workspace ─────────────────────────────────────────────────────────────────────────

export function WriterWorkspace({ initialDraft = null }: { initialDraft?: Draft | null }) {
  const params = useSearchParams();
  const { state, create, review, cancel, reset } = useDraftStream(initialDraft);
  const [text, setText] = useState("");
  const [audience, setAudience] = useState<Audience>((initialDraft?.audience as Audience) ?? "grade8");
  const [scholarId, setScholarId] = useState(params.get("scholar") ?? "");
  const [documentId, setDocumentId] = useState(params.get("document") ?? "");
  const [tab, setTab] = useState<Tab>("chat");
  // On a phone one pane shows at a time: the writer until there is a story to watch, unless chosen.
  const [mobileChoice, setMobile] = useState<"writer" | "story" | null>(null);
  const thread = useRef<HTMLDivElement>(null);

  const sections = useMemo(() => viewSections(state), [state]);
  const allSegments = sections.flatMap((s) => s.segments);
  const numbers = numberCitations(allSegments);
  const passages = state.draft?.sources ?? state.sources;
  const checks = checksFor(sections);
  const status = agentStatus(state);
  const streaming = state.status === "streaming";
  const started = state.thread.length > 0;
  const showOutline = sections.length > 0;
  const wordCount = words(allSegments);
  const mobile = mobileChoice ?? (state.outline || state.draft ? "story" : "writer");

  // The URL follows the draft, so a reload or a shared link opens it.
  useEffect(() => {
    if (state.draftId && (state.status === "review" || state.status === "published") && !location.pathname.endsWith(state.draftId)) {
      window.history.replaceState(null, "", `/write/${encodeURIComponent(state.draftId)}`);
    }
  }, [state.draftId, state.status]);

  useEffect(() => {
    const el = thread.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [state.thread.length, streaming]);

  const send = () => {
    const t = text.trim();
    if (!t || streaming) return;
    if (state.status === "review" && state.draftId) {
      void review(state.draftId, { decision: "revise", notes: t });
    } else {
      void create({ brief: t, audience, scholar_id: scholarId || null, document_id: documentId || null });
    }
    setText("");
    setTab("chat");
  };

  const newStory = () => {
    reset();
    setText("");
    setTab("chat");
    setMobile(null);
    window.history.replaceState(null, "", "/write");
  };

  const approve = () => {
    if (state.draftId) void review(state.draftId, { decision: "approve", notes: null });
  };

  const goTo = (i: number) => {
    setMobile("story");
    requestAnimationFrame(() => document.getElementById(`story-sec-${i}`)?.scrollIntoView({ behavior: "smooth", block: "start" }));
  };

  const pickIdea = (idea: string) => {
    setText(idea);
    setTab("chat");
    setMobile("writer");
    requestAnimationFrame(() => document.getElementById("writer-input")?.focus());
  };

  const composerMode = streaming ? "busy" : state.status === "review" ? "revise" : "brief";
  const canStart = state.status === "idle" || (state.status === "error" && !state.draftId);
  const pill = PILL[state.status];
  const meta =
    state.status === "idle"
      ? "Nothing written yet"
      : streaming && state.outline
        ? `${state.sections.filter((s) => s.status === "accepted" || s.status === "dropped").length} of ${state.outline.sections.length} sections done`
        : streaming
          ? "Researching"
          : wordCount
            ? `${wordCount.toLocaleString("en-US")} words · ${Math.max(1, Math.round(wordCount / 200))} min read`
            : "";

  const tabs: { key: Tab; label: string; badge?: number }[] = [
    { key: "chat", label: "Chat" },
    { key: "ideas", label: "Ideas" },
    { key: "checks", label: "Checks", badge: checks.length || undefined },
    { key: "sources", label: "Sources", badge: passages.length || undefined },
  ];

  return (
    <div className="flex h-[calc(100dvh-3.5rem-1px)] flex-col" data-chat-page="">
      {/* One bar: where you are, what state the story is in, and the one decision that matters now. */}
      <div className="flex min-h-12 flex-wrap items-center gap-x-3 gap-y-1 border-b border-rule px-4 py-1.5">
        <DraftsMenu onNew={newStory} />
        <span className={`rounded-full border px-2.5 py-0.5 text-xs font-medium ${pill.cls}`}>{pill.text}</span>
        <span className="hidden truncate text-xs text-muted sm:inline">{meta}</span>
        <div className="ml-auto flex items-center gap-2">
          <div className="flex rounded-lg border border-rule p-0.5 text-xs lg:hidden" role="tablist" aria-label="View">
            {(["writer", "story"] as const).map((v) => (
              <button
                key={v}
                type="button"
                role="tab"
                aria-selected={mobile === v}
                onClick={() => setMobile(v)}
                className={`rounded-md px-2.5 py-1 ${mobile === v ? "bg-ink text-paper" : "text-muted"}`}
              >
                {v === "writer" ? "Writer" : "Story"}
              </button>
            ))}
          </div>
          {state.status === "review" && (
            <button type="button" onClick={approve} className="inline-flex h-8 items-center rounded-lg bg-ink px-3.5 text-sm font-medium text-paper hover:opacity-90">
              Approve &amp; publish
            </button>
          )}
          {state.status === "published" && (
            <button type="button" onClick={newStory} className="inline-flex h-8 items-center rounded-lg border border-rule bg-surface px-3.5 text-sm font-medium hover:bg-sunk">
              New story
            </button>
          )}
        </div>
      </div>

      <div
        className={`grid min-h-0 flex-1 ${showOutline ? "lg:grid-cols-[13rem_minmax(0,1fr)_25rem]" : "lg:grid-cols-[minmax(0,1fr)_25rem]"}`}
      >
        {showOutline && (
          <nav aria-label="Outline" className="hidden min-h-0 overflow-y-auto border-r border-rule px-3 py-5 lg:block">
            <p className="mb-2 flex items-center justify-between px-2 font-mono text-xs uppercase tracking-[0.14em] text-muted">
              Outline <span>{sections.length}</span>
            </p>
            <ol className="space-y-0.5">
              {sections.map((s, i) => (
                <li key={s.key}>
                  <button type="button" onClick={() => goTo(i)} className="flex w-full items-start gap-2 rounded-lg px-2 py-1.5 text-left text-sm hover:bg-sunk">
                    <span className="pt-1.5">
                      <Dot tone={sectionTone(s)} />
                    </span>
                    <span className={s.status === "pending" ? "text-muted" : ""}>{s.heading}</span>
                  </button>
                </li>
              ))}
            </ol>
            <div className="mt-4 space-y-1 px-2 text-xs text-muted">
              <p className="flex items-center gap-2"><Dot tone="busy" /> being written</p>
              <p className="flex items-center gap-2"><Dot tone="ok" /> supported</p>
              <p className="flex items-center gap-2"><Dot tone="warn" /> needs a look</p>
              <p className="flex items-center gap-2"><Dot tone="bad" /> dropped</p>
            </div>
          </nav>
        )}

        <div role="region" className={`min-h-0 overflow-y-auto ${mobile === "story" ? "block" : "hidden"} lg:block`} aria-label="Story" aria-live="polite">
          <Canvas state={state} sections={sections} audience={audience} onIdea={pickIdea} />
        </div>

        <aside aria-label={WRITER} className={`min-h-0 flex-col border-l border-rule bg-surface ${mobile === "writer" ? "flex" : "hidden"} lg:flex`}>
          <div className="flex items-center gap-3 px-4 pb-2 pt-3">
            <WriterMark busy={streaming} />
            <div className="min-w-0">
              <p className="text-sm font-semibold">{WRITER}</p>
              <p
                role="status"
                className={`truncate text-xs ${
                  status.tone === "warn" ? "text-warn" : status.tone === "ok" ? "text-ok" : status.tone === "bad" ? "text-bad" : "text-muted"
                }`}
              >
                {status.text}
              </p>
            </div>
          </div>
          <div role="tablist" aria-label="Writer sections" className="flex gap-0.5 border-b border-rule px-2">
            {tabs.map((t) => (
              <button
                key={t.key}
                type="button"
                role="tab"
                aria-selected={tab === t.key}
                onClick={() => setTab(t.key)}
                className={`relative inline-flex items-center gap-1.5 px-2.5 py-2 text-xs transition ${
                  tab === t.key ? "text-ink after:absolute after:inset-x-1.5 after:bottom-0 after:h-0.5 after:rounded-full after:bg-ink" : "text-muted hover:text-ink"
                }`}
              >
                {icons[t.key]}
                {t.label}
                {t.badge ? (
                  <span className={`rounded-full px-1.5 text-[10px] font-semibold leading-4 ${t.key === "checks" ? "bg-warn-soft text-warn" : "bg-sunk text-muted"}`}>
                    {t.badge}
                  </span>
                ) : null}
              </button>
            ))}
          </div>

          <div ref={tab === "chat" ? thread : undefined} className="min-h-0 flex-1 overflow-y-auto px-4 py-4">
            {tab === "chat" && <Thread state={state} />}
            {tab === "ideas" && (
              <div className="space-y-3">
                <p className="text-sm text-muted">
                  {state.status === "review"
                    ? "Things to ask for in a revision. Tap one to put it in the chat, then change it before you send."
                    : canStart
                      ? "Ways this story could start. Tap one to put it in the chat, then change it before you send."
                      : "Ideas appear when the writer is waiting for you."}
                </p>
                {(state.status === "review" || canStart) && (
                  <ul className="space-y-2">
                    {(state.status === "review" ? REVISIONS : STARTERS).map((s) => (
                      <li key={s}>
                        <button type="button" onClick={() => pickIdea(s)} className="w-full rounded-xl border border-rule bg-paper px-3.5 py-2.5 text-left text-sm leading-snug hover:border-accent/40 hover:bg-sunk">
                          {s}
                        </button>
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            )}
            {tab === "checks" && <ChecksTab sections={sections} state={state} onGo={goTo} />}
            {tab === "sources" && <SourcesTab passages={passages} numbers={numbers} />}
          </div>

          {state.status !== "published" && (
            <Composer mode={composerMode} value={text} setValue={setText} onSend={send} onStop={cancel}>
              {canStart && !started ? (
                <Setup
                  audience={audience}
                  setAudience={setAudience}
                  scholarId={scholarId}
                  setScholarId={setScholarId}
                  documentId={documentId}
                  setDocumentId={setDocumentId}
                />
              ) : null}
            </Composer>
          )}
        </aside>
      </div>
    </div>
  );
}
