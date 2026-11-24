# API contract

The one place the backend (FastAPI, `backend/`), the frontend (Next.js, `frontend/`) and the fine-tuning
pipeline (`finetune/`) agree on shapes. JSON keys are `snake_case` everywhere. Base URL in development:
`http://localhost:8000`. Every route is under `/api`, except the MCP endpoint at `/mcp`.

When `FOOTNOTE_API_KEYS` is configured, agent routes, draft routes, and `/mcp` require
`Authorization: Bearer <key>`. Keys map to stable owner ids; draft list/get/review operations are
restricted to the owner. Collection read routes remain public.

When a shape here changes, change it here first.

---

## Shared objects

```ts
type License = {
  id: "CC-BY-4.0" | "CC-BY-SA-4.0" | "CC0-1.0" | "PD-US";
  name: string;            // "Creative Commons Attribution-ShareAlike 4.0"
  url: string;             // the licence text
  attribution: string;     // the line we must show next to anything derived from the source
};

type Scholar = {
  id: string;              // slug, e.g. "w-e-b-du-bois"
  name: string;
  born: number | null;     // year
  died: number | null;
  summary: string;         // 1-3 sentences, from the Wikipedia lead
  kind: "historical" | "author";   // historical = public-domain era; author = wrote a CC-BY paper
  document_count: number;
  wikipedia_url: string | null;
};

type DocumentSummary = {
  id: string;              // e.g. "gutenberg-408", "wikipedia-w-e-b-du-bois", "pmc-PMC1234567"
  title: string;
  kind: "biography" | "work" | "paper";
  year: number | null;
  scholar_ids: string[];
  source_url: string;
  license: License;
  chunk_count: number;
};

type Passage = {           // one retrieved chunk
  id: string;              // "<document_id>:<ordinal 4 digits>", e.g. "gutenberg-408:0042"
  document_id: string;
  title: string;           // document title
  section: string | null;  // nearest heading
  text: string;
  scholar_ids: string[];
  license_id: License["id"];
  source_url: string;
  attribution: string;     // the source licence's attribution line; show it wherever the passage is shown
  score: number | null;    // retrieval or rerank score, when it came from a search
};

type Citation = {
  passage_id: string;
  cited_text: string;      // exact text Claude cited (from native search_result citations)
};

type Verdict = {           // the grounding judge's ruling on one claim
  label: "supported" | "partial" | "unsupported";
  evidence: string;        // a verbatim quote from the passages, "" when unsupported
  rationale: string;       // one sentence
  judge: string;           // which model ruled, e.g. "claude-sonnet-5-5" or "local:footnote-judge"
};

type Segment = {           // one sentence-or-so of an answer or a draft section
  text: string;
  citations: Citation[];
  verdict: Verdict | null; // null = framing text that states no fact (nothing to judge), or not judged yet
};

type Usage = { input_tokens: number; output_tokens: number; cost_usd: number };
```

## Server-sent events

Streaming routes answer `text/event-stream`. Each event has an `event:` name and a JSON `data:` line.
Every stream ends with `done` or `error`. Common events:

| event | data |
|---|---|
| `step` | `{ "node": string, "status": "start" \| "done", "detail": string \| null }`: a graph node started or finished |
| `token` | `{ "text": string }`: streamed answer text (display only; the final segments replace it) |
| `error` | `{ "message": string }` |
| `done` | `{ "run_id": string, "usage": Usage, "latency_ms": number, "trace_url": string \| null }` |

---

## Routes

### `GET /api/health`
`{ "status": "ok", "offline": boolean, "models": { "<route>": "<model id>" }, "corpus": { "scholars": n, "documents": n, "chunks": n } }`
`offline: true` means fakes are serving (no API keys), so the UI can say so.

### `GET /api/scholars` → `Scholar[]`
### `GET /api/scholars/{id}` → `{ "scholar": Scholar, "documents": DocumentSummary[] }`
### `GET /api/documents/{id}` → `DocumentSummary & { "sections": string[] }`
### `GET /api/passages/{id}` → `Passage`

### `POST /api/ask`, the RAG agent (SSE)
Request: `{ "question": string, "scholar_id": string | null, "strategy": "bm25" | "dense" | "hybrid" | "hybrid_rerank", "history": { "role": "user" | "assistant", "content": string }[] }`
(`strategy` defaults to `hybrid_rerank`; the others exist for the ablation and the UI's "retrieval" toggle.
`history` is optional: the chat's earlier turns, oldest first, at most 12. The planner uses it to rewrite a
follow-up such as "what about Woodson?" into a standalone question; retrieval and the answer use only that
rewritten question, so an earlier answer is never treated as a source.)

Events, in order: `step`* → `passages` → `token`* → `step`* → `answer` → `done`.

| event | data |
|---|---|
| `passages` | `{ "question": string, "queries": string[], "passages": Passage[] }`: the standalone question (the follow-up rewritten from the conversation; the question itself otherwise), the sub-queries and the reranked passages the answer may cite |
| `verdicts` | `{ "checked": number, "failing": number }`: after each judging pass (a revision follows when `failing > 0` and budget is left) |
| `answer` | `{ "segments": Segment[], "removed": Segment[], "revised": boolean, "abstained": boolean }`: final, judged answer. `removed` = segments the judge still found unsupported after the revision budget; they are taken out of the answer and listed so the UI can show what was withheld. `abstained` = the sources do not answer the question |

Note on `token`: an event with `"reset": true` means a new answer attempt started (a revision), so the UI clears the streamed text.

### `POST /api/drafts`, the writer agent (SSE, pauses for review)
Request: `{ "brief": string, "audience": "grade4" | "grade8" | "adult", "scholar_id": string | null, "document_id": string | null }`

Events: `step`* → `draft` (once, with the id) → `sources` → `outline` → `section`* → `readability` → `review` → `done`.
The stream ends at `review`, because the graph is paused at a human-in-the-loop `interrupt()`.

| event | data |
|---|---|
| `draft` | `{ "draft_id": string }` |
| `sources` | `{ "passages": Passage[] }`: what the research agent picked (the draft can cite only these) |
| `readability` | `{ "grade_level": number, "target": string, "editor_note": string, "fit": "fits" \| "too_hard" \| "too_easy", "retrying": boolean }`: after each drafting round. `retrying` = a misfit within the retry budget, so one more round follows with the note as guidance; a misfit with `retrying: false` goes to review as it is |
| `outline` | `{ "title": string, "sections": { "heading": string, "goal": string, "passage_ids": string[] }[] }` |
| `section` | `{ "index": number, "heading": string, "attempt": number, "status": "drafting" \| "judging" \| "accepted" \| "redrafting" \| "dropped", "segments": Segment[] }` |
| `review` | `{ "draft": Draft }`: waiting for a human decision |
| `published` | `{ "draft": Draft }` |

### `POST /api/drafts/{id}/review` (SSE, resumes the graph)
Request: `{ "decision": "approve" | "revise", "notes": string | null }`
`404` if there is no such owned draft; `409` if the draft is not `awaiting_review` or another
review request atomically claimed it first.
`approve` → `published` → `done`. `revise` → the graph redrafts using the notes → `section`* → `review` → `done`.

### `GET /api/drafts` → `DraftSummary[]` and `GET /api/drafts/{id}` → `Draft`
```ts
type DraftSummary = { id: string; title: string | null; audience: string; status: "running" | "awaiting_review" | "published" | "failed"; created_at: string; updated_at: string };
type Draft = DraftSummary & {
  brief: string;
  sections: { heading: string; segments: Segment[]; warnings: string[] }[];
  readability: { grade_level: number; target: string; editor_note: string } | null;
  sources: Passage[];
};
```

### `POST /api/chat`, the supervisor agent (SSE)
A chat agent that calls the other agents and the MCP tools. Request: `{ "messages": { "role": "user" | "assistant", "content": string }[] }`

| event | data |
|---|---|
| `tool_call` | `{ "id": string, "name": string, "input": object }`: e.g. `ask_corpus`, `search_passages`, `start_draft` |
| `tool_result` | `{ "id": string, "name": string, "summary": string, "is_error": boolean }` |
| `token` | `{ "text": string }` |
| `message` | `{ "content": string, "citations": Citation[] }`: the final assistant message. When it relays an `ask_corpus` answer, its `[n]` markers map to `citations[n-1]` (in marker order; `cited_text` is `""` because the relay quotes the answer, not the passage) |

### `GET /api/evals` → latest evaluation results (written by `backend/evals/`)
```ts
type EvalReport = {
  generated_at: string | null;   // null until an eval has been run
  retrieval: { strategy: string; n: number; recall_at_5: number; recall_at_10: number; mrr: number; ndcg_at_10: number }[];
  answers: { config: string; n: number; faithfulness: number; citation_precision: number; answer_relevance: number; abstention_accuracy: number; p50_latency_ms: number; cost_per_answer_usd: number }[];
  judge: { judge: string; n: number; macro_f1: number; unsupported_recall: number; json_valid_rate: number; p50_latency_ms: number; cost_per_1k_usd: number }[];
};
```
Empty arrays until an eval has been run.

Only a complete, non-smoke `footnote eval all` run may replace `latest.json`. It records corpus,
model, and dataset provenance. `footnote eval check` applies `config/eval-gates.yaml` and exits
nonzero for stale, incomplete, unreviewed, undersized, or below-threshold production results.

### `/mcp`, Model Context Protocol (Streamable HTTP)
The same tools the agents use, for any MCP client (Claude Code, Claude Desktop, the supervisor agent).
Tools: `search_passages`, `get_passage`, `list_scholars`, `get_scholar`, `ask_corpus`.
Resources: `footnote://scholars/{id}`, `footnote://documents/{id}`. Prompt: `write_story`.
Also runs over stdio: `uv run footnote-mcp`.

---

## The grounding judge (shared by the agents and `finetune/`)

Input: one claim and the passages it cites. Output (JSON, exactly these keys):

```json
{ "label": "supported" | "partial" | "unsupported", "evidence": "<verbatim quote or empty>", "rationale": "<one sentence>" }
```

- `supported`: every part of the claim is stated in, or follows directly from, the passages.
- `partial`: the core is supported but the claim adds a detail, number, scope or certainty the passages do not give.
- `unsupported`: the passages do not say it, contradict it, or are about something else.

The fine-tuned model is trained to emit exactly this object from the prompt built by
`backend/src/footnote/judge_prompt.py::render_judge_prompt` (system + user text), so the same prompt serves
Claude and the local model and their scores are comparable. Both judges are scored with
`backend/src/footnote/judge_metrics.py::score_judge`.

Rate limits: every route that runs an agent answers `429` with a message when the per-client rate
limit or the day's spend ceiling is reached.
