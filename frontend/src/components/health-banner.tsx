"use client";

import { api, API_URL } from "@/lib/api";
import { useApi } from "@/lib/use-api";

export function CorpusCounts() {
  const health = useApi("health", (s) => api.health(s));

  if (health.status === "loading") {
    return <p className="text-sm text-muted">Checking the API…</p>;
  }
  if (health.status === "error") {
    return (
      <div role="status" className="rounded-md border border-rule bg-sunk px-3.5 py-2.5 text-sm">
        <p className="font-medium">The API isn’t reachable at {API_URL}.</p>
        <p className="text-muted">
          Start the backend, or run <code className="font-mono text-xs">npm run dev:mock</code> to see every page with
          fixture data.
        </p>
      </div>
    );
  }
  const { corpus, offline, models } = health.data;
  const stats = [
    { label: "scholars", value: corpus.scholars },
    { label: "documents", value: corpus.documents },
    { label: "passages", value: corpus.chunks },
  ];
  return (
    <div className="space-y-3">
      {offline && (
        <div role="status" className="rounded-md border border-warn/40 bg-warn-soft px-3.5 py-2.5 text-sm">
          <span className="font-semibold">Offline: fakes serving.</span> No API keys are set, so a fake model and an
          in-memory index answer instead of Claude and Pinecone. Answers show the flow, not real retrieval.
        </div>
      )}
      <dl className="flex flex-wrap gap-x-8 gap-y-2">
        {stats.map((s) => (
          <div key={s.label}>
            <dt className="text-xs uppercase tracking-[0.12em] text-muted">{s.label}</dt>
            <dd className="font-serif text-3xl font-semibold tabular-nums">{s.value.toLocaleString("en-US")}</dd>
          </div>
        ))}
      </dl>
      {Object.keys(models).length > 0 && (
        <details className="text-sm">
          <summary className="cursor-pointer text-muted hover:text-ink">Models by route</summary>
          <ul className="mt-2 grid gap-x-6 gap-y-1 font-mono text-xs sm:grid-cols-2">
            {Object.entries(models).map(([route, model]) => (
              <li key={route} className="flex justify-between gap-3 border-b border-rule/60 py-1">
                <span className="text-muted">{route}</span>
                <span>{model}</span>
              </li>
            ))}
          </ul>
        </details>
      )}
    </div>
  );
}
