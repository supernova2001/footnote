"use client";

import { useCallback, useEffect, useReducer, useRef } from "react";
import { api, errorMessage } from "./api";
import type { ChatEvent, ChatFinalMessage, ChatMessage, DoneData, ToolCallData, ToolResultData } from "./types";

export type TraceItem = { call: ToolCallData; result: ToolResultData | null };

export type ChatTurn =
  | { role: "user"; content: string }
  | {
      role: "assistant";
      status: "streaming" | "done" | "error";
      trace: TraceItem[];
      text: string; // streamed tokens, replaced by `final` when it arrives
      final: ChatFinalMessage | null;
      done: DoneData | null;
      error: string | null;
    };

type Action =
  | { type: "send"; content: string }
  | { type: "event"; e: ChatEvent }
  | { type: "fail"; message: string }
  | { type: "ended" }
  | { type: "reset" };

function updateLast(turns: ChatTurn[], f: (t: Extract<ChatTurn, { role: "assistant" }>) => ChatTurn): ChatTurn[] {
  const last = turns[turns.length - 1];
  if (!last || last.role !== "assistant") return turns;
  return [...turns.slice(0, -1), f(last)];
}

function reducer(turns: ChatTurn[], action: Action): ChatTurn[] {
  switch (action.type) {
    case "reset":
      return [];
    case "send":
      return [
        ...turns,
        { role: "user", content: action.content },
        { role: "assistant", status: "streaming", trace: [], text: "", final: null, done: null, error: null },
      ];
    case "fail":
      return updateLast(turns, (t) => ({ ...t, status: "error", error: action.message }));
    case "ended":
      return updateLast(turns, (t) =>
        t.status === "streaming"
          ? t.final
            ? { ...t, status: "done" }
            : { ...t, status: "error", error: "The stream ended before the reply was finished." }
          : t,
      );
    case "event": {
      const { e } = action;
      switch (e.event) {
        case "step":
          return turns;
        case "tool_call":
          return updateLast(turns, (t) => ({ ...t, trace: [...t.trace, { call: e.data, result: null }] }));
        case "tool_result":
          return updateLast(turns, (t) => ({
            ...t,
            trace: t.trace.map((item) => (item.call.id === e.data.id ? { ...item, result: e.data } : item)),
          }));
        case "token":
          return updateLast(turns, (t) => ({ ...t, text: t.text + e.data.text }));
        case "message":
          return updateLast(turns, (t) => ({ ...t, final: e.data }));
        case "done":
          return updateLast(turns, (t) => ({ ...t, status: "done", done: e.data }));
        case "error":
          return updateLast(turns, (t) => ({ ...t, status: "error", error: e.data.message }));
      }
    }
  }
}

function toMessages(turns: ChatTurn[]): ChatMessage[] {
  const out: ChatMessage[] = [];
  for (const t of turns) {
    if (t.role === "user") out.push({ role: "user", content: t.content });
    else {
      const content = t.final?.content ?? t.text;
      if (t.status === "done" && content) out.push({ role: "assistant", content });
    }
  }
  return out;
}

export function useChat() {
  const [turns, dispatch] = useReducer(reducer, []);
  const controller = useRef<AbortController | null>(null);
  const turnsRef = useRef(turns);
  useEffect(() => {
    turnsRef.current = turns;
  });
  useEffect(() => () => controller.current?.abort(), []);

  const send = useCallback(async (content: string) => {
    controller.current?.abort();
    const ac = new AbortController();
    controller.current = ac;
    const messages = [...toMessages(turnsRef.current), { role: "user" as const, content }];
    dispatch({ type: "send", content });
    try {
      for await (const e of api.chat({ messages }, ac.signal)) {
        if (ac.signal.aborted) return;
        dispatch({ type: "event", e });
      }
      if (!ac.signal.aborted) dispatch({ type: "ended" });
    } catch (err) {
      if (!ac.signal.aborted) dispatch({ type: "fail", message: errorMessage(err) });
    }
  }, []);

  const stop = useCallback(() => {
    controller.current?.abort();
    dispatch({ type: "fail", message: "Stopped." });
  }, []);

  const reset = useCallback(() => {
    controller.current?.abort();
    dispatch({ type: "reset" });
  }, []);

  return { turns, send, stop, reset };
}
