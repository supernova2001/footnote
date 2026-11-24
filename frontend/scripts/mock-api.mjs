// A dependency-free mock of the Footnote API (docs/api-contract.md) for frontend work
// without the backend. Every response is contract-shaped; the content is invented
// placeholder data (see mock-fixtures.mjs).
//
//   node scripts/mock-api.mjs          # http://localhost:8787
//   MOCK_PORT=9000 MOCK_SPEED=0 ...    # speed 0 = no delays (tests); 2 = twice as slow
//   MOCK_EMPTY_EVALS=1 ...             # /api/evals returns the empty report
//
// Questions containing "error" make /api/ask fail mid-stream; "weather" or "mars" make it abstain.

import { randomUUID } from "node:crypto";
import { createServer } from "node:http";
import { pathToFileURL } from "node:url";
import * as fx from "./mock-fixtures.mjs";

const SPEED = Number(process.env.MOCK_SPEED ?? "1");
const sleep = (ms) => new Promise((r) => setTimeout(r, Math.max(0, ms * SPEED)));

const CORS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type, Accept",
};

const drafts = new Map([[fx.seededDraft.id, structuredClone(fx.seededDraft)]]);

function json(res, status, body) {
  res.writeHead(status, { "Content-Type": "application/json; charset=utf-8", ...CORS });
  res.end(JSON.stringify(body));
}

function notFound(res, what = "Not found") {
  json(res, 404, { detail: what });
}

async function readBody(req) {
  const chunks = [];
  for await (const c of req) chunks.push(c);
  if (chunks.length === 0) return {};
  try {
    return JSON.parse(Buffer.concat(chunks).toString("utf8"));
  } catch {
    return null;
  }
}

/** Open an SSE response; returns send(event, data) and an `open` flag that drops when the client leaves. */
function sse(req, res) {
  res.writeHead(200, {
    "Content-Type": "text/event-stream; charset=utf-8",
    "Cache-Control": "no-cache, no-transform",
    Connection: "keep-alive",
    "X-Accel-Buffering": "no",
    ...CORS,
  });
  const state = { open: true };
  req.on("close", () => (state.open = false));
  res.on("close", () => (state.open = false));
  const send = (event, data) => {
    if (!state.open) return false;
    res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);
    return true;
  };
  const end = () => {
    if (state.open) res.end();
    state.open = false;
  };
  return { send, end, state };
}

function usage(inTok, outTok) {
  return { input_tokens: inTok, output_tokens: outTok, cost_usd: Number(((inTok * 4 + outTok * 20) / 1e6).toFixed(5)) };
}

function done(send, started, inTok, outTok) {
  send("done", { run_id: randomUUID(), usage: usage(inTok, outTok), latency_ms: Date.now() - started, trace_url: null });
}

async function streamWords(send, state, text, { reset = false, delay = 35 } = {}) {
  const words = text.split(/(\s+)/);
  let first = true;
  for (const w of words) {
    if (!state.open) return;
    if (!w) continue;
    send("token", first && reset ? { text: w, reset: true } : { text: w });
    first = false;
    await sleep(w.trim() ? delay : 0);
  }
}

async function step(send, node, detail = null, ms = 250, doneDetail = undefined) {
  send("step", { node, status: "start", detail });
  await sleep(ms);
  send("step", { node, status: "done", detail: doneDetail === undefined ? detail : doneDetail });
}

// ── /api/ask ────────────────────────────────────────────────────────────────

async function ask(req, res, body) {
  const question = typeof body?.question === "string" ? body.question.trim() : "";
  if (!question) return json(res, 422, { detail: "question is required" });
  const strategy = body.strategy ?? "hybrid_rerank";
  const started = Date.now();
  const { send, end, state } = sse(req, res);
  const key = fx.pickAnswer(question);
  const a = fx.answers[key];

  await step(send, "guard", null, 200, key === "abstain" ? "in scope: a question for the corpus" : "in scope");
  await step(send, "rewrite", null, 300, `${a.queries.length} sub-queries`);

  if (/\berror\b/i.test(question)) {
    send("step", { node: "retrieve", status: "start", detail: null });
    await sleep(300);
    send("error", { message: "Mock failure: the vector index timed out (ask without the word “error” to get an answer)." });
    return end();
  }

  const label = { bm25: "BM25", dense: "dense", hybrid: "BM25 + dense, RRF", hybrid_rerank: "BM25 + dense, RRF" }[strategy] ?? strategy;
  await step(send, "retrieve", null, 400, `24 candidates · ${label}`);
  if (strategy === "hybrid_rerank") await step(send, "rerank", null, 350, `kept ${a.passageIds.length}`);
  const scored = a.passageIds.map(([id, score], i) => fx.withScore(id, strategy === "bm25" ? Number((score * 0.9 - i * 0.03).toFixed(3)) : score));
  send("passages", { queries: a.queries, passages: scored });

  send("step", { node: "answer", status: "start", detail: null });
  await streamWords(send, state, a.firstDraft);
  send("step", { node: "answer", status: "done", detail: null });

  for (let i = 0; i < a.passes.length && state.open; i++) {
    const pass = a.passes[i];
    send("step", { node: "verify", status: "start", detail: null });
    await sleep(600);
    send("verdicts", pass);
    send("step", { node: "verify", status: "done", detail: `${pass.checked} claims · ${pass.failing} failing` });
    const last = i === a.passes.length - 1;
    if (!last && pass.failing > 0) {
      send("step", { node: "revise", status: "start", detail: `naming ${pass.failing} failing claim${pass.failing === 1 ? "" : "s"}` });
      await streamWords(send, state, a.revisedDraft ?? a.firstDraft, { reset: true });
      send("step", { node: "revise", status: "done", detail: null });
    }
  }
  if (!state.open) return;

  await step(send, "finalize", null, 150, a.removed.length ? `${a.removed.length} claim withheld` : null);
  send("answer", { segments: a.segments, removed: a.removed, revised: a.revised, abstained: key === "abstain" });
  done(send, started, 5200, 410);
  end();
}

// ── /api/drafts ─────────────────────────────────────────────────────────────

function sectionsFrom(story, { revisedSection = null } = {}) {
  return story.sections.map((s, i) => ({
    heading: story.outline[i].heading,
    segments: i === 1 ? s.second : s.first,
    warnings: i === 1 && s.warning ? [s.warning] : i === revisedSection ? ["Redrafted from your notes."] : [],
  }));
}

function makeDraft(body) {
  const now = new Date().toISOString();
  const story = fx.draftStory;
  return {
    id: `d-${randomUUID().slice(0, 8)}`,
    title: story.title,
    audience: body.audience ?? "grade4",
    status: "running",
    created_at: now,
    updated_at: now,
    brief: body.brief,
    sections: [],
    readability: null,
    sources: story.sources.map((id) => fx.findPassage(id)),
  };
}

async function sendSection(send, index, attempt, status, segments, ms = 350) {
  send("section", { index, heading: fx.draftStory.outline[index].heading, attempt, status, segments });
  await sleep(ms);
}

const unjudged = (segs) => segs.map((s) => ({ ...s, verdict: null }));

function readabilityFor(target) {
  const grade = { grade4: 4.6, grade8: 7.9, adult: 10.4 }[target] ?? 4.6;
  const label = { grade4: "grade 4", grade8: "grade 8", adult: "adult" }[target] ?? target;
  return { grade_level: grade, target, editor_note: `On target for ${label} readers; sentences average 12 words. ${fx.MOCK_NOTE}` };
}

async function createDraft(req, res, body) {
  if (!body || typeof body.brief !== "string" || !body.brief.trim()) return json(res, 422, { detail: "brief is required" });
  const started = Date.now();
  const { send, end, state } = sse(req, res);
  const draft = makeDraft(body);
  drafts.set(draft.id, draft);
  const story = fx.draftStory;

  await step(send, "research", body.document_id ? `limited to ${body.document_id}` : null, 500, `picked ${story.sources.length} passages from 2 documents`);
  send("draft", { draft_id: draft.id });
  await step(send, "outline", null, 400, `${story.outline.length} sections`);
  send("outline", { title: story.title, sections: story.outline });

  send("step", { node: "draft", status: "start", detail: `${story.outline.length} sections in parallel` });
  for (let i = 0; i < story.outline.length; i++) await sendSection(send, i, 1, "drafting", [], 150);
  await sleep(500);
  const s = story.sections;
  await sendSection(send, 0, 1, "judging", unjudged(s[0].first));
  await sendSection(send, 1, 1, "judging", unjudged(s[1].first));
  await sendSection(send, 0, 1, "accepted", s[0].first);
  await sendSection(send, 2, 1, "judging", unjudged(s[2].first));
  await sendSection(send, 1, 1, "redrafting", s[1].first, 500);
  await sendSection(send, 2, 1, "accepted", s[2].first);
  await sendSection(send, 1, 2, "drafting", [], 600);
  await sendSection(send, 1, 2, "judging", unjudged(s[1].second), 500);
  await sendSection(send, 1, 2, "accepted", s[1].second);
  send("step", { node: "draft", status: "done", detail: "1 section redrafted" });
  if (!state.open) return;

  await step(send, "readability", null, 400, `grade ${readabilityFor(body.audience ?? "grade4").grade_level} for ${body.audience ?? "grade4"}`);
  send("step", { node: "human_review", status: "start", detail: "waiting for you" });

  Object.assign(draft, {
    status: "awaiting_review",
    sections: sectionsFrom(story),
    readability: readabilityFor(body.audience ?? "grade4"),
    updated_at: new Date().toISOString(),
  });
  send("review", { draft });
  done(send, started, 14800, 2100);
  end();
}

async function reviewDraft(req, res, id, body) {
  const draft = drafts.get(id);
  if (!draft) return notFound(res, "No such draft");
  if (!body || (body.decision !== "approve" && body.decision !== "revise")) return json(res, 422, { detail: "decision must be approve or revise" });
  if (draft.status !== "awaiting_review") return json(res, 409, { detail: `Draft is ${draft.status}, not awaiting review` });
  const started = Date.now();
  const { send, end } = sse(req, res);

  if (body.decision === "approve") {
    await step(send, "human_review", null, 150, "approved");
    await step(send, "publish", null, 300);
    Object.assign(draft, { status: "published", updated_at: new Date().toISOString() });
    send("published", { draft });
    done(send, started, 0, 0);
    return end();
  }

  await step(send, "human_review", null, 150, `changes requested: “${String(body.notes ?? "").slice(0, 60)}”`);
  send("step", { node: "draft", status: "start", detail: "redrafting section 1 from your notes" });
  const s = fx.draftStory.sections[0].first;
  await sendSection(send, 0, 2, "drafting", [], 500);
  await sendSection(send, 0, 2, "judging", unjudged(s), 500);
  await sendSection(send, 0, 2, "accepted", s);
  send("step", { node: "draft", status: "done", detail: null });
  send("step", { node: "human_review", status: "start", detail: "waiting for you" });
  Object.assign(draft, { sections: sectionsFrom(fx.draftStory, { revisedSection: 0 }), updated_at: new Date().toISOString() });
  send("review", { draft });
  done(send, started, 6100, 820);
  end();
}

function summary(d) {
  const { id, title, audience, status, created_at, updated_at } = d;
  return { id, title, audience, status, created_at, updated_at };
}

// ── /api/chat ───────────────────────────────────────────────────────────────

async function chat(req, res, body) {
  const messages = Array.isArray(body?.messages) ? body.messages : null;
  const last = messages?.filter((m) => m.role === "user").at(-1);
  if (!last || typeof last.content !== "string") return json(res, 422, { detail: "messages must end with a user message" });
  const started = Date.now();
  const { send, end, state } = sse(req, res);
  const text = last.content;
  const key = fx.pickAnswer(text);
  const a = fx.answers[key];

  send("step", { node: "supervisor", status: "start", detail: null });
  send("tool_call", { id: "call_1", name: "search_passages", input: { query: a.queries[0], k: 5 } });
  await sleep(700);
  send("tool_result", { id: "call_1", name: "search_passages", summary: `${a.passageIds.length} passages from ${new Set(a.passageIds.map(([id]) => id.split(":")[0])).size} documents`, is_error: false });

  send("tool_call", { id: "call_2", name: "ask_corpus", input: { question: text, scholar_id: null } });
  await sleep(1100);
  const supportedCount = a.segments.filter((s) => s.verdict?.label === "supported").length;
  send("tool_result", {
    id: "call_2",
    name: "ask_corpus",
    summary: key === "abstain" ? "Abstained: the sources don't cover this" : `Answer with ${supportedCount} supported claims${a.removed.length ? `, ${a.removed.length} withheld` : ""}`,
    is_error: false,
  });

  let draftLine = "";
  if (/(story|draft|write)/i.test(text)) {
    const d = makeDraft({ brief: text, audience: "grade4" });
    Object.assign(d, { status: "awaiting_review", sections: sectionsFrom(fx.draftStory), readability: readabilityFor("grade4") });
    drafts.set(d.id, d);
    send("tool_call", { id: "call_3", name: "start_draft", input: { brief: text, audience: "grade4" } });
    await sleep(900);
    send("tool_result", { id: "call_3", name: "start_draft", summary: `Draft ${d.id} is waiting for review`, is_error: false });
    draftLine = `\n\nI started a draft for you; it is waiting for your review at /write/${d.id}.`;
  }

  const content =
    (key === "abstain"
      ? "I searched the corpus and asked the RAG agent, and neither found anything on this. The sources here are historical scholarship, so I'd rather say so than guess."
      : a.segments.map((s) => s.text).join(" ")) + draftLine;
  send("step", { node: "supervisor", status: "done", detail: null });
  await streamWords(send, state, content, { delay: 25 });
  const citations = a.segments.flatMap((s) => s.citations);
  send("message", { content, citations });
  done(send, started, 9100, 640);
  end();
}

// ── Router ──────────────────────────────────────────────────────────────────

async function handle(req, res) {
  const url = new URL(req.url ?? "/", "http://localhost");
  const path = decodeURIComponent(url.pathname).replace(/\/+$/, "") || "/";
  if (req.method === "OPTIONS") {
    res.writeHead(204, CORS);
    return res.end();
  }

  if (req.method === "GET") {
    if (path === "/api/health")
      return json(res, 200, {
        status: "ok",
        offline: process.env.MOCK_OFFLINE !== "0",
        models: {
          fixtures: "mock-api.mjs (invented placeholder content)",
          "ask.answer": "claude-opus-5-5",
          "judge.grounding": "claude-sonnet-5-5",
          "writer.draft": "claude-opus-5-5",
          "supervisor.chat": "claude-opus-5-5",
        },
        corpus: { scholars: fx.scholars.length, documents: fx.documents.length, chunks: fx.documents.reduce((n, d) => n + d.chunk_count, 0) },
      });
    if (path === "/api/scholars") return json(res, 200, fx.scholars);
    let m = path.match(/^\/api\/scholars\/(.+)$/);
    if (m) {
      const scholar = fx.scholars.find((s) => s.id === m[1]);
      if (!scholar) return notFound(res, "No such scholar");
      // DocumentSummary has no `sections`; GET /api/documents/{id} adds them.
      const docs = fx.documents
        .filter((d) => d.scholar_ids.includes(scholar.id))
        .map((d) => {
          const rest = { ...d };
          delete rest.sections;
          return rest;
        });
      return json(res, 200, { scholar, documents: docs });
    }
    m = path.match(/^\/api\/documents\/(.+)$/);
    if (m) {
      const d = fx.documents.find((x) => x.id === m[1]);
      return d ? json(res, 200, d) : notFound(res, "No such document");
    }
    m = path.match(/^\/api\/passages\/(.+)$/);
    if (m) {
      const p = fx.findPassage(m[1]);
      return p ? json(res, 200, p) : notFound(res, "No such passage");
    }
    if (path === "/api/drafts") return json(res, 200, [...drafts.values()].map(summary).sort((a, b) => b.updated_at.localeCompare(a.updated_at)));
    m = path.match(/^\/api\/drafts\/([^/]+)$/);
    if (m) {
      const d = drafts.get(m[1]);
      return d ? json(res, 200, d) : notFound(res, "No such draft");
    }
    if (path === "/api/evals") return json(res, 200, process.env.MOCK_EMPTY_EVALS === "1" ? fx.emptyEvalReport : fx.evalReport);
    if (path === "/") return json(res, 200, { mock: true, see: "docs/api-contract.md" });
  }

  if (req.method === "POST") {
    const body = await readBody(req);
    if (body === null) return json(res, 400, { detail: "Body is not valid JSON" });
    if (path === "/api/ask") return ask(req, res, body);
    if (path === "/api/drafts") return createDraft(req, res, body);
    if (path === "/api/chat") return chat(req, res, body);
    const m = path.match(/^\/api\/drafts\/([^/]+)\/review$/);
    if (m) return reviewDraft(req, res, m[1], body);
  }

  return notFound(res);
}

export function startMock(port = Number(process.env.MOCK_PORT ?? "8787")) {
  const server = createServer((req, res) => {
    handle(req, res).catch((e) => {
      console.error("[mock-api]", e);
      if (!res.headersSent) json(res, 500, { detail: String(e?.message ?? e) });
      else res.end();
    });
  });
  return new Promise((resolve) => {
    server.listen(port, () => {
      console.log(`[mock-api] Footnote mock API on http://localhost:${port} (fixtures are invented placeholder content)`);
      resolve(server);
    });
  });
}

if (import.meta.url === pathToFileURL(process.argv[1] ?? "").href) {
  startMock();
}
