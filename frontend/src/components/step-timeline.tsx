import { STRATEGY_LABEL } from "@/lib/format";
import type { StepEntry } from "@/lib/steps";

const NODE_LABEL: Record<string, string> = {
  guard: "Scope check",
  rewrite: "Rewrite query",
  retrieve: "Retrieve",
  rerank: "Rerank",
  answer: "Answer",
  verify: "Judge claims",
  revise: "Revise",
  finalize: "Finalize",
  research: "Research",
  pick_sources: "Pick sources",
  outline: "Outline",
  draft: "Draft sections",
  draft_section: "Draft section",
  write_section: "Write section",
  review: "Your review",
  plan: "Plan queries",
  judge: "Judge",
  judge_section: "Judge section",
  assemble: "Assemble",
  readability: "Reading level",
  readability_edit: "Reading level",
  human_review: "Human review",
  publish: "Publish",
  supervisor: "Supervisor",
};

export function nodeLabel(node: string): string {
  return NODE_LABEL[node] ?? node.replace(/_/g, " ");
}

/** The graph's progress as an ordered list. Each node shows a dot: pulsing while running, filled when done. */
export function StepTimeline({ steps, title = "Agent steps" }: { steps: StepEntry[]; title?: string }) {
  if (steps.length === 0) return null;
  return (
    <section aria-label={title}>
      <h2 className="mb-2 font-mono text-xs uppercase tracking-[0.14em] text-muted">{title}</h2>
      <ol className="relative space-y-2 border-l border-rule pl-4">
        {steps.map((s, i) => (
          <li key={`${s.node}-${i}`} className="relative text-sm">
            <span
              aria-hidden="true"
              className={`absolute -left-[21px] top-1.5 h-2.5 w-2.5 rounded-full border-2 ${
                s.status === "done" ? "border-ink bg-ink" : "pulse border-accent bg-paper"
              }`}
            />
            <span className={s.status === "done" ? "text-ink" : "text-accent"}>{nodeLabel(s.node)}</span>
            <span className="sr-only">{s.status === "done" ? " (done)" : " (running)"}</span>
            {s.detail && <span className="block text-xs text-muted">{STRATEGY_LABEL[s.detail] ?? s.detail}</span>}
          </li>
        ))}
      </ol>
    </section>
  );
}
