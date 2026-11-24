"use client";

import { useEffect, useState } from "react";
import { api, errorMessage } from "./api";
import type { Passage } from "./types";

const cache = new Map<string, Passage>();

/** A passage by id: from the list the stream already sent, else fetched once and cached. */
export function usePassage(id: string | null, known: Passage[]): { passage: Passage | null; error: string | null } {
  const local = id ? (known.find((p) => p.id === id) ?? cache.get(id) ?? null) : null;
  const [fetched, setFetched] = useState<{ id: string; passage: Passage | null; error: string | null } | null>(null);

  useEffect(() => {
    if (!id || local) return;
    const ac = new AbortController();
    api.passage(id, ac.signal).then(
      (p) => {
        cache.set(id, p);
        setFetched({ id, passage: p, error: null });
      },
      (e: unknown) => {
        if (!ac.signal.aborted) setFetched({ id, passage: null, error: errorMessage(e) });
      },
    );
    return () => ac.abort();
  }, [id, local]);

  if (!id) return { passage: null, error: null };
  if (local) return { passage: local, error: null };
  if (fetched && fetched.id === id) return { passage: fetched.passage, error: fetched.error };
  return { passage: null, error: null };
}
