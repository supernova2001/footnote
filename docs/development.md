# Development guide

How to run Footnote locally and how to run its evaluations. For what the system is and how it is
designed, see the [README](../README.md) and [architecture.md](architecture.md).

## Run it

**No keys needed (offline mode).** Fakes stand in for Claude, Pinecone and MongoDB. The bundled
corpus snapshot is real public-domain text (Ida B. Wells and Carter G. Woodson).

```bash
cd backend && uv sync && FOOTNOTE_OFFLINE=1 uv run footnote-api      # http://localhost:8000
cd frontend && npm ci && npm run dev                                  # http://localhost:3000
```

The UI also runs on its own with `npm run dev:mock`, against a contract-shaped mock API.

**Real models, no database yet.** Set `FOOTNOTE_STORAGE=local` and a model key in `.env`
(`FOOTNOTE_LLM_PROVIDER=anthropic` with `ANTHROPIC_API_KEY`, or `openai` with `OPENAI_API_KEY`;
`openai` uses the cheapest tiers, gpt-6-luna and gpt-5-nano, while `openai-quality` uses the mid and
top tiers). The agents then run on real models over the bundled snapshot, kept in memory. Retrieval
here is a stand-in (BM25 plus a hashed-vector proxy), so answer quality is what the real index gives.

**Full stack.** Copy `.env.example` to `.env` and fill in a model key plus Pinecone and MongoDB
(local or Atlas M0). LangSmith is optional.

```bash
docker compose up mongo -d
cd backend && uv run footnote ingest --snapshot --export-chunks ../finetune/data/chunks.jsonl
uv run footnote-api
```

`footnote ingest` fetches the corpus in [`backend/config/corpus.yaml`](../backend/config/corpus.yaml),
checks each licence, chunks the text, stores it in MongoDB and indexes it in Pinecone (creating the
index on first run). Wikipedia requires a contact in the User-Agent: set `FOOTNOTE_FETCH_CONTACT` to
your project URL. The papers chosen are frozen in `corpus.lock.json`, so the corpus can be rebuilt
exactly. `docker compose up --build` runs all three services.

`footnote import-researchers --source-db <database>` imports openly licensed papers from a
researcher-profile export held in MongoDB. It re-verifies every source's licence, writes its outputs
to `backend/data/researcher-*.json` (ignored by git), and imports researchers by name only.

## Use Footnote from Claude Code (MCP)

```bash
claude mcp add footnote -- uv --directory backend run footnote-mcp       # stdio
claude mcp add --transport http footnote http://localhost:8000/mcp       # or the running API
```

## Evaluation

```bash
uv run footnote eval build --limit 120     # model-written questions from sampled passages
uv run footnote eval review                # keep / edit / drop each one by hand
uv run footnote eval retrieval             # bm25 vs dense vs hybrid vs hybrid+rerank
uv run footnote eval answers --langsmith   # full agent, graded by a different model; LangSmith experiment
uv run footnote eval judge --judges api local cascade
uv run footnote eval all --langsmith       # only a complete, non-smoke run updates latest.json
uv run footnote eval check                 # CI/release gate from config/eval-gates.yaml
```

Complete production results are written to `backend/evals/results/latest.json`, which the app shows
at `/evals`. Partial, smoke and offline runs use separate files and cannot replace it. The release
gates are in [`backend/config/eval-gates.yaml`](../backend/config/eval-gates.yaml).

Fine-tuning the judge (the Kaggle/Colab T4 run, the dataset design, and GGUF + Ollama serving) is
covered in [finetune/README.md](../finetune/README.md).

## Quality gates

CI runs ruff, mypy and pytest for the backend, with the repository contract tests run against a real
MongoDB service. The fine-tune pipeline has its own tests. The frontend is linted, typechecked and
built. All of it runs without API keys.
