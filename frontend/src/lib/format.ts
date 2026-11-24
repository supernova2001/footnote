import type { Citation, LicenseId, Scholar, Segment, VerdictLabel } from "./types";

export function lifespan(s: Pick<Scholar, "born" | "died">): string {
  if (s.born && s.died) return `${s.born}–${s.died}`;
  if (s.born) return `b. ${s.born}`;
  if (s.died) return `d. ${s.died}`;
  return "";
}

export const LICENSE_SHORT: Record<LicenseId, string> = {
  "CC-BY-4.0": "CC BY 4.0",
  "CC-BY-SA-4.0": "CC BY-SA 4.0",
  "CC0-1.0": "CC0",
  "PD-US": "Public domain (US)",
};

export const LICENSE_NAME: Record<LicenseId, string> = {
  "CC-BY-4.0": "Creative Commons Attribution 4.0",
  "CC-BY-SA-4.0": "Creative Commons Attribution-ShareAlike 4.0",
  "CC0-1.0": "Creative Commons Zero 1.0",
  "PD-US": "Public domain in the United States",
};

export const VERDICT_TEXT: Record<VerdictLabel, string> = {
  supported: "Supported",
  partial: "Partly supported",
  unsupported: "Not supported",
};

export const VERDICT_GLYPH: Record<VerdictLabel, string> = {
  supported: "✓",
  partial: "◐",
  unsupported: "✕",
};

export const STRATEGY_LABEL: Record<string, string> = {
  bm25: "Keyword (BM25)",
  dense: "Dense",
  hybrid: "Hybrid (RRF)",
  hybrid_rerank: "Hybrid + rerank",
};

export const AUDIENCE_LABEL: Record<string, string> = {
  grade4: "Grade 4",
  grade8: "Grade 8",
  adult: "Adult",
};

/** Number citations in order of first appearance: the footnote numbers a reader sees. */
export function numberCitations(segments: Segment[]): Map<string, number> {
  const numbers = new Map<string, number>();
  for (const s of segments) {
    for (const c of s.citations) {
      if (!numbers.has(c.passage_id)) numbers.set(c.passage_id, numbers.size + 1);
    }
  }
  return numbers;
}

export function numberCitationList(citations: Citation[]): Map<string, number> {
  const numbers = new Map<string, number>();
  for (const c of citations) if (!numbers.has(c.passage_id)) numbers.set(c.passage_id, numbers.size + 1);
  return numbers;
}

export function verdictTally(segments: Segment[]): Record<VerdictLabel | "pending", number> {
  const tally = { supported: 0, partial: 0, unsupported: 0, pending: 0 };
  for (const s of segments) {
    if (s.verdict) tally[s.verdict.label] += 1;
    else tally.pending += 1;
  }
  return tally;
}

export function pct(n: number): string {
  return `${(n * 100).toFixed(1)}%`;
}

export function num(n: number, digits = 3): string {
  return n.toFixed(digits);
}

export function usd(n: number): string {
  if (n === 0) return "$0";
  if (n < 0.01) return `$${n.toFixed(4)}`;
  return `$${n.toFixed(2)}`;
}

export function ms(n: number): string {
  if (n >= 1000) return `${(n / 1000).toFixed(1)} s`;
  return `${Math.round(n)} ms`;
}

export function shortDate(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toISOString().slice(0, 16).replace("T", " ") + " UTC";
}
