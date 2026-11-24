"use client";

import { useEffect, useRef, useState } from "react";
import { errorMessage } from "./api";

export type Loadable<T> =
  | { status: "loading" }
  | { status: "ok"; data: T }
  | { status: "error"; error: string };

/**
 * Load JSON on the client. `key` decides when to reload (e.g. the scholar id);
 * the loader itself may change on every render without causing refetches.
 */
export function useApi<T>(key: string, load: (signal: AbortSignal) => Promise<T>): Loadable<T> & { reload: () => void } {
  const loader = useRef(load);
  useEffect(() => {
    loader.current = load;
  });
  const [nonce, setNonce] = useState(0);
  const fullKey = `${key}#${nonce}`;
  const [result, setResult] = useState<{ key: string; value: Loadable<T> } | null>(null);

  useEffect(() => {
    const ac = new AbortController();
    loader.current(ac.signal).then(
      (data) => setResult({ key: fullKey, value: { status: "ok", data } }),
      (e: unknown) => {
        if (!ac.signal.aborted) setResult({ key: fullKey, value: { status: "error", error: errorMessage(e) } });
      },
    );
    return () => ac.abort();
  }, [fullKey]);

  const value: Loadable<T> = result && result.key === fullKey ? result.value : { status: "loading" };
  return { ...value, reload: () => setNonce((n) => n + 1) };
}
