"use client";

import type { ReactNode } from "react";
import { api } from "@/lib/api";
import { STRATEGY_LABEL, ms, num, pct, shortDate, usd } from "@/lib/format";
import type { AnswerRow, JudgeRow, RetrievalRow } from "@/lib/types";
import { useApi } from "@/lib/use-api";
import { Button, EmptyState, Notice, Spinner } from "./ui";

type Column<R> = { key: string; label: string; title?: string; render: (r: R) => ReactNode; best?: (rows: R[]) => number };

function bestIndex<R>(rows: R[], value: (r: R) => number, higherIsBetter = true): number {
  let best = -1;
  rows.forEach((r, i) => {
    if (best === -1) best = i;
    else if (higherIsBetter ? value(r) > value(rows[best]) : value(r) < value(rows[best])) best = i;
  });
  return best;
}

function Table<R>({ caption, rows, columns, rowKey }: { caption: string; rows: R[]; columns: Column<R>[]; rowKey: (r: R) => string }) {
  const bests = columns.map((c) => (c.best ? c.best(rows) : -1));
  return (
    // `relative` keeps the visually-hidden labels (absolutely positioned) inside the scroll box.
    <div className="relative overflow-x-auto rounded-lg border border-rule bg-surface">
      <table className="w-full min-w-[36rem] text-sm">
        <caption className="sr-only">{caption}</caption>
        <thead>
          <tr className="border-b border-rule text-left text-xs uppercase tracking-[0.08em] text-muted">
            {columns.map((c) => (
              <th key={c.key} scope="col" className="px-3 py-2.5 font-medium" title={c.title}>
                {c.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={rowKey(r)} className="border-b border-rule/60 last:border-0">
              {columns.map((c, j) => (
                <td
                  key={c.key}
                  className={`px-3 py-2.5 tabular-nums ${j === 0 ? "font-medium" : "font-mono text-[13px]"} ${
                    bests[j] === i && rows.length > 1 ? "font-semibold text-ok" : ""
                  }`}
                >
                  {c.render(r)}
                  {bests[j] === i && rows.length > 1 && <span className="sr-only"> (best)</span>}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

const RETRIEVAL: Column<RetrievalRow>[] = [
  { key: "strategy", label: "Strategy", render: (r) => STRATEGY_LABEL[r.strategy] ?? r.strategy },
  { key: "n", label: "n", render: (r) => r.n },
  { key: "r5", label: "Recall@5", render: (r) => num(r.recall_at_5), best: (rows) => bestIndex(rows, (r) => r.recall_at_5) },
  { key: "r10", label: "Recall@10", render: (r) => num(r.recall_at_10), best: (rows) => bestIndex(rows, (r) => r.recall_at_10) },
  { key: "mrr", label: "MRR", title: "Mean reciprocal rank", render: (r) => num(r.mrr), best: (rows) => bestIndex(rows, (r) => r.mrr) },
  { key: "ndcg", label: "nDCG@10", render: (r) => num(r.ndcg_at_10), best: (rows) => bestIndex(rows, (r) => r.ndcg_at_10) },
];

const ANSWERS: Column<AnswerRow>[] = [
  { key: "config", label: "Config", render: (r) => r.config },
  { key: "n", label: "n", render: (r) => r.n },
  { key: "faith", label: "Faithfulness", render: (r) => pct(r.faithfulness), best: (rows) => bestIndex(rows, (r) => r.faithfulness) },
  { key: "cite", label: "Citation precision", render: (r) => pct(r.citation_precision), best: (rows) => bestIndex(rows, (r) => r.citation_precision) },
  { key: "rel", label: "Relevance", render: (r) => pct(r.answer_relevance), best: (rows) => bestIndex(rows, (r) => r.answer_relevance) },
  { key: "abst", label: "Abstention acc.", title: "Abstains exactly when the sources can't answer", render: (r) => pct(r.abstention_accuracy), best: (rows) => bestIndex(rows, (r) => r.abstention_accuracy) },
  { key: "lat", label: "p50 latency", render: (r) => ms(r.p50_latency_ms), best: (rows) => bestIndex(rows, (r) => r.p50_latency_ms, false) },
  { key: "cost", label: "Cost / answer", render: (r) => usd(r.cost_per_answer_usd), best: (rows) => bestIndex(rows, (r) => r.cost_per_answer_usd, false) },
];

const JUDGE: Column<JudgeRow>[] = [
  { key: "judge", label: "Judge", render: (r) => <span className="font-mono text-[13px]">{r.judge}</span> },
  { key: "n", label: "n", render: (r) => r.n },
  { key: "f1", label: "Macro-F1", render: (r) => num(r.macro_f1), best: (rows) => bestIndex(rows, (r) => r.macro_f1) },
  { key: "ur", label: "Unsupported recall", title: "Share of unsupported claims the judge caught", render: (r) => pct(r.unsupported_recall), best: (rows) => bestIndex(rows, (r) => r.unsupported_recall) },
  { key: "json", label: "Valid JSON", render: (r) => pct(r.json_valid_rate), best: (rows) => bestIndex(rows, (r) => r.json_valid_rate) },
  { key: "lat", label: "p50 latency", render: (r) => ms(r.p50_latency_ms), best: (rows) => bestIndex(rows, (r) => r.p50_latency_ms, false) },
  { key: "cost", label: "Cost / 1k", render: (r) => usd(r.cost_per_1k_usd), best: (rows) => bestIndex(rows, (r) => r.cost_per_1k_usd, false) },
];

const EVAL_STEPS: [string, string][] = [
  ["footnote eval build", "draft questions from the corpus"],
  ["footnote eval review", "check and label them by hand"],
  ["footnote eval retrieval", "each retrieval strategy: recall, MRR, nDCG"],
  ["footnote eval answers", "faithfulness, citations, abstention"],
  ["footnote eval judge", "the fine-tuned judge against a frontier judge"],
];

function Section({ id, title, children, intro }: { id: string; title: string; intro: string; children: ReactNode }) {
  return (
    <section aria-labelledby={id} className="space-y-3">
      <div>
        <h2 id={id} className="font-serif text-2xl font-semibold">
          {title}
        </h2>
        <p className="mt-1 max-w-3xl text-sm text-muted">{intro}</p>
      </div>
      {children}
    </section>
  );
}

export function EvalTables() {
  const report = useApi("evals", (s) => api.evals(s));

  if (report.status === "loading") return <Spinner label="Loading results" />;
  if (report.status === "error")
    return (
      <div className="space-y-3">
        <Notice tone="bad">{report.error}</Notice>
        <Button variant="secondary" onClick={report.reload}>
          Try again
        </Button>
      </div>
    );

  const { retrieval, answers, judge, generated_at } = report.data;
  const empty = retrieval.length + answers.length + judge.length === 0;

  if (empty) {
    return (
      <EmptyState title="No evaluation has been run yet">
        <p>From the backend, build the question set, review it by hand, then run the suites:</p>
        <ol className="mx-auto mt-3 max-w-md space-y-1.5 text-left">
          {EVAL_STEPS.map(([cmd, what]) => (
            <li key={cmd} className="flex flex-col gap-0.5 sm:flex-row sm:items-baseline sm:gap-3">
              <code className="shrink-0 rounded bg-sunk px-1.5 py-0.5 font-mono text-xs text-ink">{cmd}</code>
              <span className="text-xs">{what}</span>
            </li>
          ))}
        </ol>
        <p className="mt-3">
          Each suite writes <code className="font-mono">backend/evals/results/latest.json</code>, which this page reads
          from <code className="font-mono">GET /api/evals</code>.
        </p>
      </EmptyState>
    );
  }

  return (
    <div className="space-y-10">
      <p className="text-xs text-muted">
        {generated_at ? `Generated ${shortDate(generated_at)}. ` : ""}Best value per column in green.
      </p>
      <Section id="ev-retrieval" title="Retrieval" intro="Ablation over retrieval strategies on labelled questions: does each added stage (dense, fusion, rerank) earn its latency?">
        {retrieval.length ? (
          <Table caption="Retrieval results" rows={retrieval} columns={RETRIEVAL} rowKey={(r) => r.strategy} />
        ) : (
          <EmptyState title="Not run yet" />
        )}
      </Section>
      <Section id="ev-answers" title="Answers" intro="End-to-end answers, scored for faithfulness to the cited passages, citation precision, relevance, and abstaining exactly when the sources are silent.">
        {answers.length ? <Table caption="Answer results" rows={answers} columns={ANSWERS} rowKey={(r) => r.config} /> : <EmptyState title="Not run yet" />}
      </Section>
      <Section id="ev-judge" title="Grounding judge" intro="The QLoRA fine-tuned model against its base model and a frontier-model judge, on the same prompt and a hand-labelled test set.">
        {judge.length ? <Table caption="Judge results" rows={judge} columns={JUDGE} rowKey={(r) => r.judge} /> : <EmptyState title="Not run yet" />}
      </Section>
    </div>
  );
}
