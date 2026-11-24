"use client";

import { useCallback, useEffect, useReducer, useRef } from "react";
import { api, errorMessage } from "./api";
import { closeSteps, foldStep, type StepEntry } from "./steps";
import type {
  CreateDraftRequest,
  DoneData,
  Draft,
  DraftEvent,
  OutlineData,
  Passage,
  ReadabilityData,
  ReviewRequest,
  SectionData,
  SectionStatus,
} from "./types";

export type SectionState = SectionData & { history: { attempt: number; status: SectionStatus }[] };

/** One line of the conversation with the writer: what the person said, or what the agent did. */
export type ThreadEntry =
  | { kind: "user"; text: string }
  | { kind: "sources"; count: number; works: number }
  | { kind: "outline"; title: string; headings: string[] }
  | { kind: "section"; index: number; heading: string; status: "accepted" | "redrafting" | "dropped"; attempt: number }
  | { kind: "readability"; grade: number; target: string; note: string; fit: ReadabilityData["fit"]; retrying: boolean }
  | { kind: "review" }
  | { kind: "published" }
  | { kind: "error"; message: string };

export type DraftStreamState = {
  status: "idle" | "streaming" | "review" | "published" | "error";
  draftId: string | null;
  steps: StepEntry[];
  sources: Passage[];
  outline: OutlineData | null;
  sections: SectionState[];
  readability: ReadabilityData | null;
  draft: Draft | null; // the draft as of the last review/published event (or as loaded)
  thread: ThreadEntry[];
  done: DoneData | null;
  error: string | null;
};

const initial: DraftStreamState = {
  status: "idle",
  draftId: null,
  steps: [],
  sources: [],
  outline: null,
  sections: [],
  readability: null,
  draft: null,
  thread: [],
  done: null,
  error: null,
};

type Action =
  | { type: "start"; brief: string }
  | { type: "resume"; notes: string | null }
  | { type: "load"; draft: Draft }
  | { type: "event"; e: DraftEvent }
  | { type: "fail"; message: string }
  | { type: "ended" }
  | { type: "cancel" }
  | { type: "reset" };

const NOTED = new Set<SectionStatus>(["accepted", "redrafting", "dropped"]);

function foldSection(sections: SectionState[], s: SectionData): { sections: SectionState[]; changed: boolean } {
  const next = sections.slice();
  const prev = next[s.index];
  const history = prev ? prev.history.slice() : [];
  const last = history[history.length - 1];
  const changed = !last || last.status !== s.status || last.attempt !== s.attempt;
  if (changed) history.push({ attempt: s.attempt, status: s.status });
  // Keep the last non-empty segments while a redraft streams in with none yet.
  const segments = s.segments.length > 0 || !prev ? s.segments : prev.segments;
  next[s.index] = { ...s, segments, history };
  return { sections: next, changed };
}

function reducer(state: DraftStreamState, action: Action): DraftStreamState {
  switch (action.type) {
    case "reset":
      return initial;
    case "start":
      return { ...initial, status: "streaming", thread: [{ kind: "user", text: action.brief }] };
    case "resume":
      return {
        ...state,
        status: "streaming",
        error: null,
        done: null,
        thread: action.notes ? [...state.thread, { kind: "user", text: action.notes }] : state.thread,
      };
    case "load": {
      const d = action.draft;
      const thread: ThreadEntry[] = [{ kind: "user", text: d.brief }];
      if (d.status === "awaiting_review") thread.push({ kind: "review" });
      if (d.status === "published") thread.push({ kind: "published" });
      return {
        ...initial,
        status:
          d.status === "published" ? "published" : d.status === "awaiting_review" ? "review" : d.status === "failed" ? "error" : "idle",
        draftId: d.id,
        draft: d,
        sources: d.sources,
        readability: d.readability ? { ...d.readability, fit: "fits" } : null,
        thread,
        error: d.status === "failed" ? "This draft failed before it reached review." : null,
      };
    }
    case "fail":
      return {
        ...state,
        status: "error",
        error: action.message,
        steps: closeSteps(state.steps),
        thread: [...state.thread, { kind: "error", message: action.message }],
      };
    case "cancel":
      return state.status === "streaming"
        ? { ...state, status: "error", error: "Stopped. The draft is saved where the graph paused.", steps: closeSteps(state.steps) }
        : state;
    case "ended":
      if (state.status !== "streaming") return { ...state, steps: closeSteps(state.steps) };
      return { ...state, status: "error", error: "The stream ended before the draft reached review.", steps: closeSteps(state.steps) };
    case "event": {
      const { e } = action;
      switch (e.event) {
        case "step":
          return { ...state, steps: foldStep(state.steps, e.data) };
        case "draft":
          return { ...state, draftId: e.data.draft_id };
        case "sources": {
          const works = new Set(e.data.passages.map((p) => p.document_id)).size;
          return {
            ...state,
            sources: e.data.passages,
            thread: [...state.thread, { kind: "sources", count: e.data.passages.length, works }],
          };
        }
        case "outline":
          return {
            ...state,
            outline: e.data,
            sections: [],
            thread: [...state.thread, { kind: "outline", title: e.data.title, headings: e.data.sections.map((s) => s.heading) }],
          };
        case "section": {
          const { sections, changed } = foldSection(state.sections, e.data);
          const note: ThreadEntry[] =
            changed && NOTED.has(e.data.status)
              ? [
                  {
                    kind: "section",
                    index: e.data.index,
                    heading: e.data.heading,
                    status: e.data.status as "accepted" | "redrafting" | "dropped",
                    attempt: e.data.attempt,
                  },
                ]
              : [];
          return { ...state, sections, thread: [...state.thread, ...note] };
        }
        case "readability":
          return {
            ...state,
            readability: e.data,
            thread: [
              ...state.thread,
              {
                kind: "readability",
                grade: e.data.grade_level,
                target: e.data.target,
                note: e.data.editor_note,
                fit: e.data.fit,
                retrying: e.data.retrying ?? false,
              },
            ],
          };
        case "review":
          return {
            ...state,
            status: "review",
            draft: e.data.draft,
            draftId: e.data.draft.id,
            sources: e.data.draft.sources,
            thread: [...state.thread, { kind: "review" }],
          };
        case "published":
          return {
            ...state,
            status: "published",
            draft: e.data.draft,
            draftId: e.data.draft.id,
            thread: [...state.thread, { kind: "published" }],
          };
        case "token":
          return state;
        case "done":
          return { ...state, done: e.data, steps: closeSteps(state.steps) };
        case "error":
          return {
            ...state,
            status: "error",
            error: e.data.message,
            steps: closeSteps(state.steps),
            thread: [...state.thread, { kind: "error", message: e.data.message }],
          };
      }
    }
  }
}

export function useDraftStream(initialDraft: Draft | null = null) {
  const [state, dispatch] = useReducer(reducer, initialDraft, (d) => (d ? reducer(initial, { type: "load", draft: d }) : initial));
  const controller = useRef<AbortController | null>(null);

  useEffect(() => () => controller.current?.abort(), []);

  const consume = useCallback(async (stream: AsyncGenerator<DraftEvent>, ac: AbortController) => {
    try {
      for await (const e of stream) {
        if (ac.signal.aborted) return;
        dispatch({ type: "event", e });
      }
      if (!ac.signal.aborted) dispatch({ type: "ended" });
    } catch (err) {
      if (!ac.signal.aborted) dispatch({ type: "fail", message: errorMessage(err) });
    }
  }, []);

  const create = useCallback(
    async (req: CreateDraftRequest) => {
      controller.current?.abort();
      const ac = new AbortController();
      controller.current = ac;
      dispatch({ type: "start", brief: req.brief });
      await consume(api.createDraft(req, ac.signal), ac);
    },
    [consume],
  );

  const review = useCallback(
    async (draftId: string, req: ReviewRequest) => {
      controller.current?.abort();
      const ac = new AbortController();
      controller.current = ac;
      dispatch({ type: "resume", notes: req.decision === "revise" ? req.notes : null });
      await consume(api.review(draftId, req, ac.signal), ac);
    },
    [consume],
  );

  const cancel = useCallback(() => {
    controller.current?.abort();
    dispatch({ type: "cancel" });
  }, []);

  /** A new story: stop anything running and start from an empty page. */
  const reset = useCallback(() => {
    controller.current?.abort();
    dispatch({ type: "reset" });
  }, []);

  return { state, create, review, cancel, reset };
}
