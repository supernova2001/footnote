# Footnote: frontend

Next.js 16 (App Router, Turbopack, Cache Components) + React 19 + Tailwind CSS 4, written against
[`docs/api-contract.md`](../docs/api-contract.md). Every page fetches from the FastAPI backend in the
browser; the four agent routes stream server-sent events over `POST`.

## Run it

```bash
npm install
npm run dev:mock      # mock API on :8787 + next dev on :3000, no backend or keys needed
npm run dev           # against the real backend at NEXT_PUBLIC_API_URL (default http://localhost:8000)
```

| Script | What it does |
|---|---|
| `dev` / `dev:mock` | dev server, against the backend / against the mock API |
| `mock` | the mock API alone (`MOCK_PORT`, `MOCK_SPEED=0` for no delays, `MOCK_EMPTY_EVALS=1`, `MOCK_OFFLINE=0`) |
| `build` / `start` | production build and server. `NEXT_PUBLIC_API_URL` is inlined at build time |
| `lint` / `typecheck` | ESLint, and `next typegen && tsc --noEmit` |

The mock (`scripts/mock-api.mjs`, plain Node, no dependencies) serves every route in the contract,
including the SSE sequences: an answer that is judged, revised and has a claim withheld; an abstention
(ask about "the weather on Mars"); a failure (put "error" in the question); a draft whose second section
fails the judge and is redrafted, then paused for review; and a chat that calls three tools. Its content is
**invented placeholder data** (`scripts/mock-fixtures.mjs`): short public-domain excerpts plus text marked
`[Mock text written for UI development.]`.

## Pages

| Route | What it shows |
|---|---|
| `/` | The chat (the RAG agent), as the front page: questions as bubbles, answers with footnote markers and the judge's ruling per sentence, a live status while the agent works, withheld claims, sources and run details per answer; follow-ups carry the conversation. `/ask` redirects here |
| `/scholars`, `/scholars/[id]` | The corpus by scholar; sources with licence badges; an Ask panel limited to that scholar |
| `/write`, `/write/[id]` | The writer agent: outline, each section's drafting → judging → accepted / redrafting / dropped, then the human-in-the-loop review (approve, or request changes with notes) |
| `/chat` | The supervisor agent, with each tool call and its result inline |
| `/evals` | Retrieval ablation, answer quality, and judge comparison tables |
| `/data` | The data card: sources, licences, what is and is not collected |

## Where things are

- `src/lib/types.ts`: the contract's types. `src/lib/api.ts`: the typed client; `src/lib/sse.ts`: the SSE parser for `POST` streams.
- `src/lib/use-ask.ts`, `use-draft.ts`, `use-chat.ts`: reducers that fold each stream's events into page state.
- `src/components/citations.tsx`: the reading surface: sentences, footnote markers, verdict marks, and the margin note (beside the text when there is room, below it otherwise, decided by a container query).

## Accessibility and layout

Keyboard focus is visible everywhere; footnote markers and verdict marks are buttons with labels, and a
verdict is never shown by colour alone. Streaming regions are `aria-live="polite"` with `aria-busy`.
Motion stops under `prefers-reduced-motion`. Light and dark follow the OS, with a toggle that is applied
before first paint. Checked at 1366×607 and 390×844 with no horizontal page scroll.
