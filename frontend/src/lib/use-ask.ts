/** One question's stream folded into state: shared by every turn of the Ask chat (use-ask-chat.ts). */
import { closeSteps, foldStep, type StepEntry } from "./steps";
import type { AnswerData, AskEvent, DoneData, Passage, VerdictsData } from "./types";

export type AskState = {
  status: "idle" | "streaming" | "done" | "error";
  question: string;
  /** The question as the agent searched it (a follow-up rewritten to stand alone). */
  standalone: string;
  steps: StepEntry[];
  queries: string[];
  passages: Passage[];
  tokens: string;
  attempts: number; // answer attempts streamed so far (1 + revisions)
  passes: VerdictsData[]; // one per judging pass
  answer: AnswerData | null;
  done: DoneData | null;
  error: string | null;
};

export const initialAskState: AskState = {
  status: "idle",
  question: "",
  standalone: "",
  steps: [],
  queries: [],
  passages: [],
  tokens: "",
  attempts: 0,
  passes: [],
  answer: null,
  done: null,
  error: null,
};

export type AskAction =
  | { type: "start"; question: string }
  | { type: "event"; e: AskEvent }
  | { type: "fail"; message: string }
  | { type: "ended" }
  | { type: "cancel" };

export function askReducer(state: AskState, action: AskAction): AskState {
  switch (action.type) {
    case "start":
      return { ...initialAskState, status: "streaming", question: action.question };
    case "fail":
      return { ...state, status: "error", error: action.message, steps: closeSteps(state.steps) };
    case "cancel":
      return state.status === "streaming"
        ? { ...state, status: "error", error: "Stopped.", steps: closeSteps(state.steps) }
        : state;
    case "ended":
      if (state.status !== "streaming") return state;
      return {
        ...state,
        status: state.answer ? "done" : "error",
        error: state.answer ? null : "The stream ended before an answer arrived.",
        steps: closeSteps(state.steps),
      };
    case "event": {
      const { e } = action;
      switch (e.event) {
        case "step":
          return { ...state, steps: foldStep(state.steps, e.data) };
        case "passages":
          return { ...state, standalone: e.data.question ?? "", queries: e.data.queries, passages: e.data.passages };
        case "token":
          if (e.data.reset) return { ...state, tokens: e.data.text, attempts: state.attempts + 1 };
          return { ...state, tokens: state.tokens + e.data.text, attempts: Math.max(state.attempts, 1) };
        case "verdicts":
          return { ...state, passes: [...state.passes, e.data] };
        case "answer":
          return { ...state, answer: e.data };
        case "done":
          return { ...state, status: "done", done: e.data, steps: closeSteps(state.steps) };
        case "error":
          return { ...state, status: "error", error: e.data.message, steps: closeSteps(state.steps) };
      }
    }
  }
}
