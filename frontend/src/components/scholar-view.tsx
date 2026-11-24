"use client";

import Link from "next/link";
import { api, ApiError } from "@/lib/api";
import { lifespan } from "@/lib/format";
import { useApi } from "@/lib/use-api";
import { AskChat } from "./ask-chat";
import { LicenseBadge } from "./license-badge";
import { Button, EmptyState, Notice, Spinner, Tag } from "./ui";

const KIND_LABEL = { biography: "Biography", work: "Work", paper: "Paper" } as const;

export function ScholarView({ id }: { id: string }) {
  const detail = useApi(`scholar:${id}`, async (s) => {
    try {
      return await api.scholar(id, s);
    } catch (e) {
      if (e instanceof ApiError && e.status === 404) return null;
      throw e;
    }
  });

  if (detail.status === "loading") {
    return (
      <div className="py-10">
        <Spinner label="Loading scholar" />
      </div>
    );
  }
  if (detail.status === "error") {
    return (
      <div className="space-y-3 py-10">
        <Notice tone="bad">{detail.error}</Notice>
        <Button variant="secondary" onClick={detail.reload}>
          Try again
        </Button>
      </div>
    );
  }
  if (!detail.data) {
    return (
      <div className="py-10">
        <EmptyState title="No such scholar">
          Nothing in the corpus has the id <code className="font-mono">{id}</code>.{" "}
          <Link href="/scholars" className="underline">
            Back to all scholars
          </Link>
          .
        </EmptyState>
      </div>
    );
  }

  const { scholar, documents } = detail.data;
  return (
    <div className="space-y-10 pt-8">
      <title>{`${scholar.name} · Footnote`}</title>
      <header className="space-y-3">
        <p className="text-sm">
          <Link href="/scholars" className="text-muted hover:text-ink hover:underline">
            ← Scholars
          </Link>
        </p>
        <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
          <h1 className="font-serif text-3xl font-semibold tracking-tight sm:text-4xl">{scholar.name}</h1>
          {lifespan(scholar) && <span className="text-muted">{lifespan(scholar)}</span>}
          <Tag tone={scholar.kind === "historical" ? "neutral" : "accent"}>{scholar.kind}</Tag>
        </div>
        {scholar.summary && <p className="max-w-3xl font-serif text-lg leading-relaxed">{scholar.summary}</p>}
        {scholar.wikipedia_url && (
          <p className="text-xs text-muted">
            Summary from{" "}
            <a href={scholar.wikipedia_url} target="_blank" rel="noopener noreferrer" className="text-ink underline decoration-rule hover:decoration-accent">
              Wikipedia
            </a>
            , CC BY-SA 4.0.
          </p>
        )}
      </header>

      <section aria-labelledby="sources">
        <h2 id="sources" className="mb-3 font-serif text-2xl font-semibold">
          Sources
        </h2>
        {documents.length === 0 ? (
          <EmptyState title="No sources yet" />
        ) : (
          <ul className="divide-y divide-rule rounded-lg border border-rule bg-surface">
            {documents.map((d) => (
              <li key={d.id} className="flex flex-col gap-2 p-4 sm:flex-row sm:items-center sm:justify-between">
                <div className="min-w-0">
                  <p className="font-serif text-lg font-medium leading-snug">{d.title}</p>
                  <p className="mt-0.5 flex flex-wrap items-center gap-x-2 gap-y-1 text-xs text-muted">
                    <span>{KIND_LABEL[d.kind]}</span>
                    {d.year && <span>· {d.year}</span>}
                    <span>· {d.chunk_count} passages</span>
                    <LicenseBadge license={d.license} />
                  </p>
                  <p className="mt-1 text-xs text-muted">{d.license.attribution}</p>
                </div>
                <div className="flex shrink-0 gap-3 text-sm">
                  <a href={d.source_url} target="_blank" rel="noopener noreferrer" className="text-muted underline decoration-rule hover:text-ink hover:decoration-accent">
                    Source ↗
                  </a>
                  <Link
                    href={`/write?document=${encodeURIComponent(d.id)}&scholar=${encodeURIComponent(scholar.id)}`}
                    className="font-medium text-accent hover:underline"
                  >
                    Write from this →
                  </Link>
                </div>
              </li>
            ))}
          </ul>
        )}
      </section>

      <section aria-labelledby="ask" className="border-t border-rule pt-8">
        <h2 id="ask" className="mb-1 font-serif text-2xl font-semibold">
          Ask about {scholar.name}
        </h2>
        <p className="mb-4 text-sm text-muted">Retrieval is limited to this scholar’s sources.</p>
        <AskChat scholarId={scholar.id} scholarName={scholar.name} variant="embedded" />
      </section>
    </div>
  );
}
