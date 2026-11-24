"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { api, errorMessage } from "./api";
import type { ChatMessage, Segment, Strategy } from "./types";
import { askReducer, initialAskState, type AskAction, type AskState } from "./use-ask";

export type AskTurn = { id: string; state: AskState };

/** Prior exchanges the planner may see. Small on purpose: enough to resolve "what about him?". */
const HISTORY_TURNS = 4;
const HISTORY_CHARS = 1500;

function answerText(segments: Segment[]): string {
  const text = segments.map((s) => s.text.trim()).join(" ");
  return text.length > HISTORY_CHARS ? `${text.slice(0, HISTORY_CHARS)}…` : text;
}

/** Answered turns as chat history, oldest first. Failed or stopped turns are left out. */
export function historyFrom(turns: AskTurn[]): ChatMessage[] {
  return turns
    .filter((t) => t.state.answer && t.state.status === "done")
    .slice(-HISTORY_TURNS)
    .flatMap((t) => [
      { role: "user" as const, content: t.state.question },
      { role: "assistant" as const, content: answerText(t.state.answer!.segments) || "(no answer)" },
    ]);
}

/**
 * A conversation with the RAG agent: one `AskState` per question, each streamed by the same
 * reducer as a single ask. Follow-ups carry the earlier turns so the planner can rewrite them.
 */
export function useAskChat({ scholarId = null }: { scholarId?: string | null } = {}) {
  const [turns, setTurns] = useState<AskTurn[]>([]);
  const turnsRef = useRef<AskTurn[]>([]);
  const controller = useRef<AbortController | null>(null);

  useEffect(() => {
    turnsRef.current = turns;
  }, [turns]);
  useEffect(() => () => controller.current?.abort(), []);

  const apply = useCallback((id: string, action: AskAction) => {
    setTurns((ts) => ts.map((t) => (t.id === id ? { ...t, state: askReducer(t.state, action) } : t)));
  }, []);

  const send = useCallback(
    async (question: string, strategy: Strategy) => {
      const text = question.trim();
      if (!text) return;
      controller.current?.abort();
      const ac = new AbortController();
      controller.current = ac;
      const id = `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
      const history = historyFrom(turnsRef.current);
      setTurns((ts) => [...ts, { id, state: askReducer(initialAskState, { type: "start", question: text }) }]);
      try {
        for await (const e of api.ask({ question: text, scholar_id: scholarId, strategy, history }, ac.signal)) {
          if (ac.signal.aborted) return;
          apply(id, { type: "event", e });
        }
        if (!ac.signal.aborted) apply(id, { type: "ended" });
      } catch (err) {
        if (!ac.signal.aborted) apply(id, { type: "fail", message: errorMessage(err) });
      }
    },
    [apply, scholarId],
  );

  const stop = useCallback(() => {
    controller.current?.abort();
    const last = turnsRef.current.at(-1);
    if (last) apply(last.id, { type: "cancel" });
  }, [apply]);

  const reset = useCallback(() => {
    controller.current?.abort();
    setTurns([]);
  }, []);

  const busy = turns.at(-1)?.state.status === "streaming";
  return { turns, send, stop, reset, busy };
}
