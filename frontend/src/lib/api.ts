// Typed client for the Footnote API (docs/api-contract.md).

import { readEventStream } from "./sse";
import type {
  AskEvent,
  AskRequest,
  ChatEvent,
  ChatRequest,
  CreateDraftRequest,
  DocumentDetail,
  Draft,
  DraftEvent,
  DraftSummary,
  EvalReport,
  Health,
  Passage,
  ReviewRequest,
  Scholar,
  ScholarDetail,
} from "./types";

export * from "./types";

export const API_URL = (process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000").replace(/\/$/, "");

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number | null,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function errorFrom(res: Response): Promise<ApiError> {
  let detail = res.statusText;
  try {
    const body = await res.json();
    if (body && typeof body === "object") {
      const d = (body as { detail?: unknown; message?: unknown }).detail ?? (body as { message?: unknown }).message;
      if (typeof d === "string") detail = d;
    }
  } catch {
    // not JSON; keep statusText
  }
  return new ApiError(`${res.status} ${detail}`.trim(), res.status);
}

async function getJson<T>(path: string, signal?: AbortSignal): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_URL}${path}`, { headers: { Accept: "application/json" }, signal });
  } catch (e) {
    if (e instanceof DOMException && e.name === "AbortError") throw e;
    throw new ApiError(`Can't reach the API at ${API_URL}. Is the backend (or the mock) running?`, null);
  }
  if (!res.ok) throw await errorFrom(res);
  return (await res.json()) as T;
}

const KNOWN = {
  ask: new Set(["step", "passages", "token", "verdicts", "answer", "error", "done"]),
  draft: new Set(["step", "draft", "sources", "outline", "section", "readability", "review", "published", "token", "error", "done"]),
  chat: new Set(["step", "tool_call", "tool_result", "token", "message", "error", "done"]),
} as const;

async function* postStream<E extends { event: string }>(
  path: string,
  body: unknown,
  known: Set<string>,
  signal?: AbortSignal,
): AsyncGenerator<E> {
  let res: Response;
  try {
    res = await fetch(`${API_URL}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
      body: JSON.stringify(body),
      signal,
    });
  } catch (e) {
    if (e instanceof DOMException && e.name === "AbortError") return;
    throw new ApiError(`Can't reach the API at ${API_URL}. Is the backend (or the mock) running?`, null);
  }
  if (!res.ok) throw await errorFrom(res);
  if (!res.body) throw new ApiError("The API sent an empty stream.", res.status);
  try {
    for await (const raw of readEventStream(res.body, signal)) {
      // Unknown events are ignored so the backend can add new ones without breaking the UI.
      if (known.has(raw.event)) yield raw as unknown as E;
    }
  } catch (e) {
    if (e instanceof DOMException && e.name === "AbortError") return;
    throw e;
  }
}

export const api = {
  health: (signal?: AbortSignal) => getJson<Health>("/api/health", signal),
  scholars: (signal?: AbortSignal) => getJson<Scholar[]>("/api/scholars", signal),
  scholar: (id: string, signal?: AbortSignal) =>
    getJson<ScholarDetail>(`/api/scholars/${encodeURIComponent(id)}`, signal),
  document: (id: string, signal?: AbortSignal) =>
    getJson<DocumentDetail>(`/api/documents/${encodeURIComponent(id)}`, signal),
  passage: (id: string, signal?: AbortSignal) =>
    getJson<Passage>(`/api/passages/${encodeURIComponent(id)}`, signal),
  drafts: (signal?: AbortSignal) => getJson<DraftSummary[]>("/api/drafts", signal),
  draft: (id: string, signal?: AbortSignal) => getJson<Draft>(`/api/drafts/${encodeURIComponent(id)}`, signal),
  evals: (signal?: AbortSignal) => getJson<EvalReport>("/api/evals", signal),

  ask: (req: AskRequest, signal?: AbortSignal) => postStream<AskEvent>("/api/ask", req, KNOWN.ask, signal),
  createDraft: (req: CreateDraftRequest, signal?: AbortSignal) =>
    postStream<DraftEvent>("/api/drafts", req, KNOWN.draft, signal),
  review: (id: string, req: ReviewRequest, signal?: AbortSignal) =>
    postStream<DraftEvent>(`/api/drafts/${encodeURIComponent(id)}/review`, req, KNOWN.draft, signal),
  chat: (req: ChatRequest, signal?: AbortSignal) => postStream<ChatEvent>("/api/chat", req, KNOWN.chat, signal),
};

export function errorMessage(e: unknown): string {
  if (e instanceof ApiError) return e.message;
  if (e instanceof Error) return e.message;
  return String(e);
}
