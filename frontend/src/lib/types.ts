// Mirrors docs/api-contract.md. Change the contract first, then this file.

export type LicenseId = "CC-BY-4.0" | "CC-BY-SA-4.0" | "CC0-1.0" | "PD-US";

export type License = {
  id: LicenseId;
  name: string;
  url: string;
  attribution: string;
};

export type ScholarKind = "historical" | "author";

export type Scholar = {
  id: string;
  name: string;
  born: number | null;
  died: number | null;
  summary: string;
  kind: ScholarKind;
  document_count: number;
  wikipedia_url: string | null;
};

export type DocumentKind = "biography" | "work" | "paper";

export type DocumentSummary = {
  id: string;
  title: string;
  kind: DocumentKind;
  year: number | null;
  scholar_ids: string[];
  source_url: string;
  license: License;
  chunk_count: number;
};

export type DocumentDetail = DocumentSummary & { sections: string[] };

export type Passage = {
  id: string;
  document_id: string;
  title: string;
  section: string | null;
  text: string;
  scholar_ids: string[];
  license_id: LicenseId;
  source_url: string;
  /** The source licence's attribution line, written by the backend; show it wherever the passage is. */
  attribution: string;
  score: number | null;
};

export type Citation = {
  passage_id: string;
  cited_text: string;
};

export type VerdictLabel = "supported" | "partial" | "unsupported";

export type Verdict = {
  label: VerdictLabel;
  evidence: string;
  rationale: string;
  judge: string;
};

export type Segment = {
  text: string;
  citations: Citation[];
  /** null = framing text that states no fact (nothing to judge), or not judged yet. */
  verdict: Verdict | null;
};

export type Usage = { input_tokens: number; output_tokens: number; cost_usd: number };

export type Strategy = "bm25" | "dense" | "hybrid" | "hybrid_rerank";
export type Audience = "grade4" | "grade8" | "adult";

// ── Common SSE payloads ──────────────────────────────────────────────────────

export type StepData = { node: string; status: "start" | "done"; detail: string | null };
/** `reset: true` = a new answer attempt (a revision) started; clear the streamed text first. */
export type TokenData = { text: string; reset?: boolean };
export type ErrorData = { message: string };
export type DoneData = {
  run_id: string;
  usage: Usage;
  latency_ms: number;
  trace_url: string | null;
};

// ── Routes ───────────────────────────────────────────────────────────────────

export type Health = {
  status: "ok";
  offline: boolean;
  models: Record<string, string>;
  corpus: { scholars: number; documents: number; chunks: number };
};

export type ScholarDetail = { scholar: Scholar; documents: DocumentSummary[] };

export type AskRequest = {
  question: string;
  scholar_id: string | null;
  strategy: Strategy;
  /** Earlier turns of the conversation, oldest first (at most 12): lets the planner resolve follow-ups. */
  history?: ChatMessage[];
};

export type PassagesData = {
  /** The question as the agent searched it: a follow-up rewritten to stand alone. */
  question?: string;
  queries: string[];
  passages: Passage[];
};
/** After each judging pass; a revision follows when failing > 0 and budget is left. */
export type VerdictsData = { checked: number; failing: number };
export type AnswerData = {
  segments: Segment[];
  /** Claims still unsupported after the revision budget: taken out of the answer, listed as withheld. */
  removed: Segment[];
  revised: boolean;
  abstained: boolean;
};

export type AskEvent =
  | { event: "step"; data: StepData }
  | { event: "passages"; data: PassagesData }
  | { event: "token"; data: TokenData }
  | { event: "verdicts"; data: VerdictsData }
  | { event: "answer"; data: AnswerData }
  | { event: "error"; data: ErrorData }
  | { event: "done"; data: DoneData };

export type DraftStatus = "running" | "awaiting_review" | "published" | "failed";

export type DraftSummary = {
  id: string;
  title: string | null;
  audience: string;
  status: DraftStatus;
  created_at: string;
  updated_at: string;
};

export type DraftSection = { heading: string; segments: Segment[]; warnings: string[] };

export type Draft = DraftSummary & {
  brief: string;
  sections: DraftSection[];
  readability: { grade_level: number; target: string; editor_note: string } | null;
  sources: Passage[];
};

export type CreateDraftRequest = {
  brief: string;
  audience: Audience;
  scholar_id: string | null;
  document_id: string | null;
};

export type ReviewRequest = { decision: "approve" | "revise"; notes: string | null };

export type OutlineData = {
  title: string;
  sections: { heading: string; goal: string; passage_ids: string[] }[];
};

/** What the research agent picked: the draft can cite only these. */
export type SourcesData = { passages: Passage[] };
/** After each drafting round: the measured grade and the editor model's ruling on fit. */
export type ReadabilityData = {
  grade_level: number;
  target: string;
  editor_note: string;
  fit: "fits" | "too_hard" | "too_easy";
  /** One more drafting round follows with the note as guidance. */
  retrying?: boolean;
};

export type SectionStatus = "drafting" | "judging" | "accepted" | "redrafting" | "dropped";

export type SectionData = {
  index: number;
  heading: string;
  attempt: number;
  status: SectionStatus;
  segments: Segment[];
};

export type DraftEvent =
  | { event: "step"; data: StepData }
  | { event: "draft"; data: { draft_id: string } }
  | { event: "sources"; data: SourcesData }
  | { event: "readability"; data: ReadabilityData }
  | { event: "outline"; data: OutlineData }
  | { event: "section"; data: SectionData }
  | { event: "review"; data: { draft: Draft } }
  | { event: "published"; data: { draft: Draft } }
  | { event: "token"; data: TokenData }
  | { event: "error"; data: ErrorData }
  | { event: "done"; data: DoneData };

export type ChatMessage = { role: "user" | "assistant"; content: string };
export type ChatRequest = { messages: ChatMessage[] };

export type ToolCallData = { id: string; name: string; input: Record<string, unknown> };
export type ToolResultData = { id: string; name: string; summary: string; is_error: boolean };
export type ChatFinalMessage = { content: string; citations: Citation[] };

export type ChatEvent =
  | { event: "step"; data: StepData }
  | { event: "tool_call"; data: ToolCallData }
  | { event: "tool_result"; data: ToolResultData }
  | { event: "token"; data: TokenData }
  | { event: "message"; data: ChatFinalMessage }
  | { event: "error"; data: ErrorData }
  | { event: "done"; data: DoneData };

export type RetrievalRow = {
  strategy: string;
  n: number;
  recall_at_5: number;
  recall_at_10: number;
  mrr: number;
  ndcg_at_10: number;
};

export type AnswerRow = {
  config: string;
  n: number;
  faithfulness: number;
  citation_precision: number;
  answer_relevance: number;
  abstention_accuracy: number;
  p50_latency_ms: number;
  cost_per_answer_usd: number;
};

export type JudgeRow = {
  judge: string;
  n: number;
  macro_f1: number;
  unsupported_recall: number;
  json_valid_rate: number;
  p50_latency_ms: number;
  cost_per_1k_usd: number;
};

export type EvalReport = {
  generated_at: string;
  retrieval: RetrievalRow[];
  answers: AnswerRow[];
  judge: JudgeRow[];
};
