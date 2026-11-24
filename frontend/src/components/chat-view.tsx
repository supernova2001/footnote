"use client";

import { useEffect, useId, useRef, useState, type FormEvent, type KeyboardEvent } from "react";
import { ms, numberCitationList, usd } from "@/lib/format";
import { useChat, type ChatTurn, type TraceItem } from "@/lib/use-chat";
import { usePassage } from "@/lib/use-passage";
import type { Citation } from "@/lib/types";
import { PassageCard } from "./citations";
import { Button, Notice, inputClass } from "./ui";

const EXAMPLES = [
  "Which scholars in the corpus wrote about education?",
  "Find what Ida B. Wells wrote about the press, then draft a short story about it for grade 8.",
  "Compare how Carter G. Woodson and Ida B. Wells used evidence.",
];

function TraceRow({ item }: { item: TraceItem }) {
  const [open, setOpen] = useState(false);
  const pending = !item.result;
  const failed = item.result?.is_error;
  return (
    <li className="text-sm">
      <button
        type="button"
        onClick={() => setOpen(!open)}
        aria-expanded={open}
        className="flex w-full items-start gap-2 rounded px-1.5 py-1 text-left hover:bg-sunk"
      >
        <span
          aria-hidden="true"
          className={`mt-1.5 h-2 w-2 shrink-0 rounded-full ${pending ? "pulse bg-accent" : failed ? "bg-bad" : "bg-ok"}`}
        />
        <span className="min-w-0">
          <span className="font-mono text-xs font-medium">{item.call.name}</span>
          <span className="sr-only">{pending ? " running" : failed ? " failed" : " finished"}</span>
          {item.result && <span className={`block text-xs ${failed ? "text-bad" : "text-muted"}`}>{item.result.summary}</span>}
        </span>
      </button>
      {open && (
        <pre className="mx-1.5 mt-1 max-h-48 overflow-auto rounded bg-sunk p-2 font-mono text-[11px] leading-relaxed">
          {JSON.stringify(item.call.input, null, 2)}
        </pre>
      )}
    </li>
  );
}

function CitationNote({ citation, n }: { citation: Citation; n: number }) {
  const { passage, error } = usePassage(citation.passage_id, []);
  if (error) return <p className="text-xs text-bad">{error}</p>;
  if (!passage) return <p className="text-xs text-muted">Loading {citation.passage_id}…</p>;
  return <PassageCard passage={passage} number={n} quotes={[citation.cited_text]} />;
}

function AssistantTurn({ turn }: { turn: Extract<ChatTurn, { role: "assistant" }> }) {
  const [openNote, setOpenNote] = useState<string | null>(null);
  const content = turn.final?.content ?? turn.text;
  const citations = turn.final?.citations ?? [];
  const numbers = numberCitationList(citations);
  const unique = [...numbers.keys()];
  const streaming = turn.status === "streaming";

  return (
    <div className="space-y-3">
      {turn.trace.length > 0 && (
        <section aria-label="Tool calls" className="rounded-md border border-rule bg-surface p-2">
          <p className="px-1.5 pb-1 font-mono text-[11px] uppercase tracking-[0.14em] text-muted">Tool calls</p>
          <ol className="space-y-0.5">
            {turn.trace.map((item) => (
              <TraceRow key={item.call.id} item={item} />
            ))}
          </ol>
        </section>
      )}
      {content ? (
        <div className={`prose-serif whitespace-pre-wrap ${streaming && !turn.final ? "caret" : ""}`}>{content}</div>
      ) : (
        streaming && <p className="text-sm text-muted">Thinking…</p>
      )}
      {unique.length > 0 && (
        <div className="space-y-2">
          <p className="flex flex-wrap items-center gap-1.5 text-xs text-muted">
            Sources
            {unique.map((pid) => (
              <button
                key={pid}
                type="button"
                className="rounded border border-rule px-1.5 py-0.5 font-sans text-xs font-semibold text-accent hover:bg-accent-soft aria-expanded:bg-accent-soft"
                aria-expanded={openNote === pid}
                onClick={() => setOpenNote(openNote === pid ? null : pid)}
              >
                {numbers.get(pid)}
              </button>
            ))}
          </p>
          {openNote && (
            <CitationNote citation={citations.find((c) => c.passage_id === openNote)!} n={numbers.get(openNote) ?? 0} />
          )}
        </div>
      )}
      {turn.error && <Notice tone="bad">{turn.error}</Notice>}
      {turn.done && (
        <p className="font-mono text-[11px] text-muted">
          {ms(turn.done.latency_ms)} · {usd(turn.done.usage.cost_usd)}
          {turn.done.trace_url && (
            <>
              {" · "}
              <a href={turn.done.trace_url} target="_blank" rel="noopener noreferrer" className="underline">
                trace ↗
              </a>
            </>
          )}
        </p>
      )}
    </div>
  );
}

export function ChatView() {
  const { turns, send, stop, reset } = useChat();
  const [draft, setDraft] = useState("");
  const uid = useId();
  const endRef = useRef<HTMLDivElement>(null);
  const last = turns[turns.length - 1];
  const streaming = last?.role === "assistant" && last.status === "streaming";

  useEffect(() => {
    endRef.current?.scrollIntoView({ block: "end", behavior: "smooth" });
  }, [turns.length]);

  const submit = (text: string) => {
    const t = text.trim();
    if (!t || streaming) return;
    setDraft("");
    void send(t);
  };

  const onSubmit = (e: FormEvent) => {
    e.preventDefault();
    submit(draft);
  };

  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit(draft);
    }
  };

  return (
    <div className="mx-auto max-w-3xl space-y-6">
      {turns.length === 0 ? (
        <div className="space-y-3">
          <p className="text-sm text-muted">
            The supervisor decides which tool to use: the RAG agent, a passage search over MCP, or the writer agent. Every
            call shows up in the trace.
          </p>
          <ul className="flex flex-col gap-2">
            {EXAMPLES.map((ex) => (
              <li key={ex}>
                <button
                  type="button"
                  onClick={() => submit(ex)}
                  className="w-full rounded-md border border-rule bg-surface px-3 py-2 text-left font-serif text-sm hover:border-accent/50 hover:bg-sunk"
                >
                  {ex}
                </button>
              </li>
            ))}
          </ul>
        </div>
      ) : (
        <ol className="space-y-6" aria-live="polite" aria-busy={streaming}>
          {turns.map((t, i) => (
            <li key={i} className={t.role === "user" ? "flex justify-end" : ""}>
              {t.role === "user" ? (
                <p className="max-w-[85%] whitespace-pre-wrap rounded-lg bg-sunk px-3.5 py-2 text-sm">
                  <span className="sr-only">You: </span>
                  {t.content}
                </p>
              ) : (
                <div>
                  <span className="sr-only">Footnote: </span>
                  <AssistantTurn turn={t} />
                </div>
              )}
            </li>
          ))}
        </ol>
      )}
      <div ref={endRef} />
      <form onSubmit={onSubmit} className="sticky bottom-3 space-y-2 rounded-lg border border-rule bg-surface p-2 shadow-sm">
        <label htmlFor={`${uid}-msg`} className="sr-only">
          Message
        </label>
        <textarea
          id={`${uid}-msg`}
          rows={2}
          className={`${inputClass} resize-none border-0 bg-transparent focus:border-0`}
          placeholder="Ask, search, or ask for a story…"
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={onKeyDown}
        />
        <div className="flex items-center justify-between gap-2 px-1">
          <Button variant="ghost" onClick={reset} disabled={turns.length === 0}>
            New chat
          </Button>
          {streaming ? (
            <Button variant="secondary" onClick={stop}>
              Stop
            </Button>
          ) : (
            <Button type="submit" disabled={!draft.trim()}>
              Send
            </Button>
          )}
        </div>
      </form>
    </div>
  );
}
