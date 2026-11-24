"use client";

import Link from "next/link";
import { useId, useState } from "react";
import { api } from "@/lib/api";
import { lifespan } from "@/lib/format";
import type { ScholarKind } from "@/lib/types";
import { useApi } from "@/lib/use-api";
import { Button, EmptyState, Notice, Spinner, Tag, inputClass } from "./ui";

const KINDS: { value: ScholarKind | "all"; label: string }[] = [
  { value: "all", label: "All" },
  { value: "historical", label: "Historical" },
  { value: "author", label: "Paper authors" },
];

export function ScholarList() {
  const scholars = useApi("scholars", (s) => api.scholars(s));
  const [query, setQuery] = useState("");
  const [kind, setKind] = useState<ScholarKind | "all">("all");
  const uid = useId();

  if (scholars.status === "loading") return <Spinner label="Loading scholars" />;
  if (scholars.status === "error")
    return (
      <div className="space-y-3">
        <Notice tone="bad">{scholars.error}</Notice>
        <Button variant="secondary" onClick={scholars.reload}>
          Try again
        </Button>
      </div>
    );

  const q = query.trim().toLowerCase();
  const shown = scholars.data.filter(
    (s) => (kind === "all" || s.kind === kind) && (!q || s.name.toLowerCase().includes(q) || s.summary.toLowerCase().includes(q)),
  );

  return (
    <div className="space-y-5">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center">
        <label htmlFor={`${uid}-q`} className="sr-only">
          Filter scholars
        </label>
        <input
          id={`${uid}-q`}
          className={`${inputClass} sm:max-w-xs`}
          placeholder="Filter by name or topic"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <div role="group" aria-label="Kind" className="flex gap-1">
          {KINDS.map((k) => (
            <button
              key={k.value}
              type="button"
              aria-pressed={kind === k.value}
              onClick={() => setKind(k.value)}
              className={`rounded-full border px-3 py-1 text-xs ${
                kind === k.value ? "border-ink bg-ink text-paper" : "border-rule text-muted hover:text-ink"
              }`}
            >
              {k.label}
            </button>
          ))}
        </div>
        <p className="text-xs text-muted sm:ml-auto" aria-live="polite">
          {shown.length} of {scholars.data.length}
        </p>
      </div>

      {shown.length === 0 ? (
        <EmptyState title={scholars.data.length === 0 ? "No scholars yet" : "No scholars match"}>
          {scholars.data.length === 0
            ? "The corpus is empty. Run the ingestion pipeline in backend/ to fetch the sources."
            : "Try a different name or clear the filter."}
        </EmptyState>
      ) : (
        <ul className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {shown.map((s) => (
            <li key={s.id}>
              <Link
                href={`/scholars/${encodeURIComponent(s.id)}`}
                className="flex h-full flex-col rounded-lg border border-rule bg-surface p-4 transition hover:border-accent/40 hover:bg-sunk"
              >
                <div className="flex items-start justify-between gap-2">
                  <h2 className="font-serif text-lg font-semibold leading-snug">{s.name}</h2>
                  <Tag tone={s.kind === "historical" ? "neutral" : "accent"}>{s.kind === "historical" ? "historical" : "author"}</Tag>
                </div>
                {lifespan(s) && <p className="text-xs text-muted">{lifespan(s)}</p>}
                {s.summary && <p className="mt-2 line-clamp-3 text-sm leading-relaxed">{s.summary}</p>}
                <p className="mt-auto pt-3 text-xs text-muted">
                  {s.document_count} source{s.document_count === 1 ? "" : "s"}
                </p>
              </Link>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
