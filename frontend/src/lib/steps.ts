import type { StepData } from "./types";

export type StepEntry = { node: string; status: "start" | "done"; detail: string | null };

/** Fold a `step` event into the ordered timeline: a start opens an entry, a done closes the latest open one. */
export function foldStep(steps: StepEntry[], step: StepData): StepEntry[] {
  if (step.status === "start") return [...steps, { node: step.node, status: "start", detail: step.detail }];
  for (let i = steps.length - 1; i >= 0; i--) {
    if (steps[i].node === step.node && steps[i].status === "start") {
      const next = steps.slice();
      next[i] = { node: step.node, status: "done", detail: step.detail ?? steps[i].detail };
      return next;
    }
  }
  return [...steps, { node: step.node, status: "done", detail: step.detail }];
}

/** Close any steps left open when a stream ends, so nothing spins forever. */
export function closeSteps(steps: StepEntry[]): StepEntry[] {
  return steps.map((s) => (s.status === "start" ? { ...s, status: "done" } : s));
}
