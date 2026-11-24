"use client";

import { useEffect, useId, useLayoutEffect, useRef, useState, type FormEvent, type KeyboardEvent, type ReactNode } from "react";
import { STRATEGY_LABEL, ms, numberCitations, usd } from "@/lib/format";
import type { Passage, Segment, Strategy } from "@/lib/types";
import type { AskState } from "@/lib/use-ask";
import { useAskChat, type AskTurn } from "@/lib/use-ask-chat";
import { PassageCard, SegmentedText, VerdictCard, VerdictSummary } from "./citations";
import { StepTimeline } from "./step-timeline";

const STRATEGIES: Strategy[] = ["hybrid_rerank", "hybrid", "dense", "bm25"];

/** What the assistant is doing, in words, while a graph node runs. */
const THINKING: Record<string, string> = {
  plan: "Planning the search",
  decline: "Checking the question",
  retrieve: "Searching the sources",
  answer: "Writing the answer",
  verify: "Checking each claim against its source",
  revise: "Rewriting flagged claims",
  finalize: "Finishing up",
};

// ── icons ─────────────────────────────────────────────────────────────────────────────────

const ICON = "h-4 w-4 shrink-0";
const ArrowUp = () => (
  <svg viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="2" className={ICON} aria-hidden="true">
    <path d="M10 16V4M4.5 9.5 10 4l5.5 5.5" strokeLinecap="round" strokeLinejoin="round" />
  </svg>
);
const StopSquare = () => (
  <svg viewBox="0 0 20 20" fill="currentColor" className="h-3.5 w-3.5" aria-hidden="true">
    <rect x="4" y="4" width="12" height="12" rx="2" />
  </svg>
);
const CopyIcon = () => (
  <svg viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.6" className={ICON} aria-hidden="true">
    <rect x="7" y="7" width="9" height="9" rx="2" />
    <path d="M13 7V5.5A1.5 1.5 0 0 0 11.5 4h-6A1.5 1.5 0 0 0 4 5.5v6A1.5 1.5 0 0 0 5.5 13H7" />
  </svg>
);
const CheckIcon = () => (
  <svg viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="2" className={ICON} aria-hidden="true">
    <path d="m5 10.5 3.2 3L15 6.5" strokeLinecap="round" strokeLinejoin="round" />
  </svg>
);
const PlusIcon = () => (
  <svg viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.8" className={ICON} aria-hidden="true">
    <path d="M10 4.5v11M4.5 10h11" strokeLinecap="round" />
  </svg>
);

/** The assistant's mark: Footnote's superscript one. */
function Avatar({ size = "sm" }: { size?: "sm" | "lg" }) {
  const box = size === "lg" ? "h-12 w-12 text-2xl" : "h-7 w-7 text-sm";
  return (
    <span
      aria-hidden="true"
      className={`inline-flex ${box} shrink-0 items-center justify-center rounded-full border border-rule bg-surface font-serif font-semibold text-accent`}
    >
      ¹
    </span>
  );
}

// ── one answer ────────────────────────────────────────────────────────────────────────────

function plainAnswer(segments: Segment[], numbers: Map<string, number>, passages: Passage[]): string {
  const body = segments
    .map((s) => s.text.trim() + [...new Set(s.citations.map((c) => c.passage_id))].map((id) => `[${numbers.get(id)}]`).join(""))
    .join(" ");
  const notes = [...numbers.entries()]
    .sort((a, b) => a[1] - b[1])
    .map(([id, n]) => {
      const p = passages.find((x) => x.id === id);
      return `[${n}] ${p ? `${p.title}${p.section ? `, ${p.section}` : ""} (${p.source_url})` : id}`;
    });
  return notes.length ? `${body}\n\n${notes.join("\n")}` : body;
}

function Thinking({ state }: { state: AskState }) {
  const running = [...state.steps].reverse().find((s) => s.status === "start");
  const label = (running && THINKING[running.node]) ?? "Thinking";
  return (
    <p className="flex items-center gap-2 text-sm text-muted" role="status">
      <span className="inline-flex gap-1" aria-hidden="true">
        {[0, 1, 2].map((i) => (
          <span key={i} className="pulse h-1.5 w-1.5 rounded-full bg-accent" style={{ animationDelay: `${i * 160}ms` }} />
        ))}
      </span>
      {label}…
    </p>
  );
}

function Withheld({ segments }: { segments: Segment[] }) {
  return (
    <details className="group mt-3 rounded-xl border border-bad/25 bg-bad-soft/40 px-3.5 py-2.5 text-sm">
      <summary className="cursor-pointer list-none text-bad marker:hidden">
        <span className="font-medium">
          {segments.length} claim{segments.length === 1 ? "" : "s"} withheld
        </span>
        <span className="text-muted"> · the judge could not match {segments.length === 1 ? "it" : "them"} to a source</span>
      </summary>
      <ul className="mt-2.5 space-y-2.5">
        {segments.map((s, i) => (
          <li key={i}>
            <p className="font-serif line-through decoration-bad/50">{s.text.trim()}</p>
            {s.verdict && (
              <div className="mt-1.5">
                <VerdictCard verdict={s.verdict} />
              </div>
            )}
          </li>
        ))}
      </ul>
    </details>
  );
}

function ActionButton({ onClick, pressed, children, label }: { onClick: () => void; pressed?: boolean; children: ReactNode; label?: string }) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={pressed}
      aria-label={label}
      className={`inline-flex items-center gap-1.5 rounded-lg px-2 py-1 text-xs transition hover:bg-sunk hover:text-ink ${
        pressed ? "bg-sunk text-ink" : "text-muted"
      }`}
    >
      {children}
    </button>
  );
}

function AssistantMessage({ turn, onRetry }: { turn: AskTurn; onRetry: (q: string) => void }) {
  const { state } = turn;
  const [panel, setPanel] = useState<"sources" | "details" | null>(null);
  const [rulings, setRulings] = useState(true);
  const [copied, setCopied] = useState(false);
  const streaming = state.status === "streaming";
  const answer = state.answer;
  const numbers = answer ? numberCitations(answer.segments) : new Map<string, number>();
  const cited = state.passages.filter((p) => numbers.has(p.id)).sort((a, b) => numbers.get(a.id)! - numbers.get(b.id)!);
  const rewritten = state.standalone && state.standalone.trim() !== state.question.trim();

  const copy = async () => {
    if (!answer) return;
    try {
      await navigator.clipboard.writeText(plainAnswer(answer.segments, numbers, state.passages));
      setCopied(true);
      setTimeout(() => setCopied(false), 1600);
    } catch {
      /* clipboard blocked: nothing to do */
    }
  };

  return (
    <div className="flex gap-3">
      <Avatar />
      <div className="min-w-0 flex-1 pt-0.5" aria-live="polite" aria-busy={streaming}>
        {rewritten && (
          <p className="mb-1.5 text-xs text-faint">
            Searched for: <span className="italic">{state.standalone}</span>
          </p>
        )}

        {!answer && streaming && !state.tokens && <Thinking state={state} />}
        {!answer && state.tokens && (
          <>
            {state.attempts > 1 && (
              <p className="mb-1.5 text-xs font-medium text-warn">A claim was flagged, so the answer is being rewritten…</p>
            )}
            <p className={`prose-serif ${streaming ? "caret" : ""}`}>{state.tokens}</p>
            {streaming && (
              <div className="mt-2">
                <Thinking state={state} />
              </div>
            )}
          </>
        )}

        {answer && (
          <>
            <SegmentedText segments={answer.segments} passages={state.passages} showVerdicts={rulings} numbers={numbers} showNotesList={false} />
            {answer.abstained && (
              <p className="mt-2 text-xs text-muted">Footnote answers only from its sources, never from memory.</p>
            )}
            {answer.removed.length > 0 && <Withheld segments={answer.removed} />}
          </>
        )}

        {state.status === "error" && (
          <div className="mt-2 flex flex-wrap items-center gap-2 rounded-xl border border-bad/30 bg-bad-soft/50 px-3 py-2 text-sm" role="alert">
            <span>{state.error}</span>
            <button type="button" className="font-medium text-bad underline decoration-bad/40 hover:decoration-bad" onClick={() => onRetry(state.question)}>
              Try again
            </button>
          </div>
        )}

        {answer && state.status === "done" && (
          <div className="mt-3 flex flex-wrap items-center gap-1">
            <ActionButton onClick={copy} label={copied ? "Copied" : "Copy answer"}>
              {copied ? <CheckIcon /> : <CopyIcon />}
              {copied ? "Copied" : "Copy"}
            </ActionButton>
            {cited.length > 0 && (
              <ActionButton onClick={() => setPanel(panel === "sources" ? null : "sources")} pressed={panel === "sources"}>
                {cited.length} source{cited.length === 1 ? "" : "s"}
              </ActionButton>
            )}
            <ActionButton onClick={() => setRulings((r) => !r)} pressed={rulings}>
              ✓ Rulings
            </ActionButton>
            <ActionButton onClick={() => setPanel(panel === "details" ? null : "details")} pressed={panel === "details"}>
              How this was answered
            </ActionButton>
            {answer.revised && <span className="ml-1 text-xs text-warn">Revised once</span>}
          </div>
        )}

        {panel === "sources" && (
          <ol className="mt-3 grid gap-2.5 sm:grid-cols-2">
            {cited.map((p) => (
              <li key={p.id}>
                <PassageCard passage={p} number={numbers.get(p.id)} compact />
              </li>
            ))}
          </ol>
        )}

        {panel === "details" && answer && (
          <div className="mt-3 space-y-4 rounded-xl border border-rule bg-sunk/50 p-4">
            <VerdictSummary segments={answer.segments} />
            <div className="grid gap-4 sm:grid-cols-2">
              <StepTimeline steps={state.steps} title="Agent steps" />
              <div className="space-y-3 text-sm">
                {state.queries.length > 0 && (
                  <section>
                    <h3 className="mb-1 font-mono text-xs uppercase tracking-[0.14em] text-muted">Searches</h3>
                    <ul className="space-y-0.5">
                      {state.queries.map((q) => (
                        <li key={q} className="font-serif italic">“{q}”</li>
                      ))}
                    </ul>
                  </section>
                )}
                {state.passes.length > 0 && (
                  <section>
                    <h3 className="mb-1 font-mono text-xs uppercase tracking-[0.14em] text-muted">Judging passes</h3>
                    <ol className="space-y-0.5">
                      {state.passes.map((p, i) => (
                        <li key={i}>
                          #{i + 1}: {p.checked} checked,{" "}
                          <span className={p.failing ? "text-bad" : "text-ok"}>{p.failing} failing</span>
                        </li>
                      ))}
                    </ol>
                  </section>
                )}
                {state.done && (
                  <p className="font-mono text-xs text-muted">
                    {ms(state.done.latency_ms)} · {state.done.usage.input_tokens.toLocaleString("en-US")} in /{" "}
                    {state.done.usage.output_tokens.toLocaleString("en-US")} out · {usd(state.done.usage.cost_usd)}
                  </p>
                )}
              </div>
            </div>
            <p className="text-xs text-muted">
              {state.passages.length} passages were retrieved; the answer could cite only those.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

function UserMessage({ text }: { text: string }) {
  return (
    <div className="flex justify-end">
      <p className="max-w-[85%] whitespace-pre-wrap rounded-3xl bg-sunk px-4 py-2.5 text-[15px] leading-relaxed">{text}</p>
    </div>
  );
}

// ── the composer ──────────────────────────────────────────────────────────────────────────

function Composer({
  busy,
  placeholder,
  onSend,
  onStop,
  strategy,
  setStrategy,
  autoFocus,
}: {
  busy: boolean;
  placeholder: string;
  onSend: (text: string) => void;
  onStop: () => void;
  strategy: Strategy;
  setStrategy: (s: Strategy) => void;
  autoFocus?: boolean;
}) {
  const [text, setText] = useState("");
  const ref = useRef<HTMLTextAreaElement>(null);
  const uid = useId();

  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`;
  }, [text]);

  const submit = (e?: FormEvent) => {
    e?.preventDefault();
    if (busy || !text.trim()) return;
    onSend(text);
    setText("");
  };

  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      submit();
    }
  };

  return (
    <form onSubmit={submit} className="mx-auto w-full max-w-3xl px-4 pb-3 pt-2">
      <div className="rounded-3xl border border-rule bg-surface px-4 pb-2.5 pt-3 shadow-[0_1px_8px_rgba(0,0,0,0.05)] transition focus-within:border-focus focus-within:ring-2 focus-within:ring-focus/15">
        <label htmlFor={`${uid}-q`} className="sr-only">
          Message
        </label>
        <textarea
          id={`${uid}-q`}
          ref={ref}
          rows={1}
          value={text}
          maxLength={500}
          autoFocus={autoFocus}
          placeholder={placeholder}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={onKeyDown}
          className="chat-input block max-h-[200px] w-full resize-none bg-transparent text-[15px] leading-relaxed text-ink placeholder:text-faint focus:outline-none"
        />
        <div className="mt-1.5 flex items-center justify-between gap-2">
          <label className="flex items-center gap-1.5 text-xs text-muted">
            <span className="sr-only sm:not-sr-only">Retrieval</span>
            <select
              value={strategy}
              onChange={(e) => setStrategy(e.target.value as Strategy)}
              className="rounded-lg border border-rule bg-paper px-2 py-1 text-xs text-ink focus:border-focus focus:outline-none"
            >
              {STRATEGIES.map((s) => (
                <option key={s} value={s}>
                  {STRATEGY_LABEL[s]}
                </option>
              ))}
            </select>
          </label>
          {busy ? (
            <button
              type="button"
              onClick={onStop}
              aria-label="Stop"
              className="inline-flex h-8 w-8 items-center justify-center rounded-full bg-ink text-paper transition hover:opacity-85"
            >
              <StopSquare />
            </button>
          ) : (
            <button
              type="submit"
              disabled={!text.trim()}
              aria-label="Send"
              className="inline-flex h-8 w-8 items-center justify-center rounded-full bg-ink text-paper transition hover:opacity-85 disabled:cursor-not-allowed disabled:opacity-30"
            >
              <ArrowUp />
            </button>
          )}
        </div>
      </div>
      <p className="mt-1.5 text-center text-[11px] leading-snug text-faint">
        Answers come only from the collection. Every sentence is checked against its source; claims that fail are withheld.
      </p>
    </form>
  );
}

// ── the chat ──────────────────────────────────────────────────────────────────────────────

export function AskChat({
  scholarId = null,
  scholarName,
  examples = [],
  variant = "page",
}: {
  scholarId?: string | null;
  scholarName?: string;
  examples?: string[];
  variant?: "page" | "embedded";
}) {
  const { turns, send, stop, reset, busy } = useAskChat({ scholarId });
  const [strategy, setStrategy] = useState<Strategy>("hybrid_rerank");
  const scroller = useRef<HTMLDivElement>(null);
  const stick = useRef(true);

  // Follow the newest message, unless the reader has scrolled up to read something older.
  useEffect(() => {
    const el = scroller.current;
    if (el && stick.current) el.scrollTop = el.scrollHeight;
  }, [turns]);

  const onScroll = () => {
    const el = scroller.current;
    if (el) stick.current = el.scrollHeight - el.scrollTop - el.clientHeight < 120;
  };

  const ask = (q: string) => {
    stick.current = true;
    void send(q, strategy);
  };

  const shell =
    variant === "page"
      ? "flex h-[calc(100dvh-3.5rem-1px)] flex-col"
      : "flex h-[38rem] max-h-[80dvh] flex-col overflow-hidden rounded-2xl border border-rule bg-paper";
  const placeholder = scholarName ? `Ask about ${scholarName}…` : "Ask about Ida B. Wells, Carter G. Woodson…";
  const empty = turns.length === 0;

  return (
    <div className={shell} data-chat-page={variant === "page" ? "" : undefined}>
      {!empty && (
        <div className="mx-auto flex w-full max-w-3xl items-center justify-end px-4 pt-2">
          <button
            type="button"
            onClick={reset}
            className="inline-flex items-center gap-1.5 rounded-lg px-2.5 py-1.5 text-xs text-muted transition hover:bg-sunk hover:text-ink"
          >
            <PlusIcon /> New chat
          </button>
        </div>
      )}

      <div ref={scroller} onScroll={onScroll} className="flex-1 overflow-y-auto">
        {empty ? (
          <div className="mx-auto flex h-full max-w-3xl flex-col items-center justify-center px-4 py-8 text-center">
            <Avatar size="lg" />
            <h1 className="mt-4 font-serif text-2xl font-semibold tracking-tight sm:text-3xl">
              {scholarName ? `What would you like to know about ${scholarName}?` : "What would you like to know?"}
            </h1>
            <p className="mt-2 max-w-md text-sm text-muted">
              I answer from the collection’s sources only. Every sentence is footnoted and checked by an independent judge.
            </p>
            {examples.length > 0 && (
              <ul className="mt-6 grid w-full max-w-2xl gap-2 sm:grid-cols-2">
                {examples.map((ex) => (
                  <li key={ex}>
                    <button
                      type="button"
                      onClick={() => ask(ex)}
                      className="h-full w-full rounded-2xl border border-rule bg-surface px-4 py-3 text-left text-sm leading-snug transition hover:border-accent/40 hover:bg-sunk"
                    >
                      {ex}
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </div>
        ) : (
          <div className="mx-auto max-w-3xl space-y-8 px-4 pb-6 pt-4">
            {turns.map((t) => (
              <div key={t.id} className="space-y-5">
                <UserMessage text={t.state.question} />
                <AssistantMessage turn={t} onRetry={ask} />
              </div>
            ))}
          </div>
        )}
      </div>

      <Composer
        busy={busy}
        placeholder={placeholder}
        onSend={ask}
        onStop={stop}
        strategy={strategy}
        setStrategy={setStrategy}
        autoFocus={variant === "page"}
      />
    </div>
  );
}
