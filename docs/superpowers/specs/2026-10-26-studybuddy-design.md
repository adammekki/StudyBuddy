# StudyBuddy — Design

Date: 2026-10-26
Status: approved in conversation, written up for planning

## Intent

A student studying for a course hits a gap while solving practice problems. Finding the answer means combing through lecture decks. Web search gives answers that are often wrong for the specific course notation and definitions. StudyBuddy answers the question from that student's own lecture slides, in a chat window, and shows which deck and page the answer came from.

The point is trust: every slide answer cites its source, and anything not from the slides is labelled as coming from the internet.

## Why now

The author just learned to build a RAG pipeline end to end with LangChain and wants a real project to test it. The course material is available as PDF exports of LaTeX beamer slides (6-10 lectures now).

## Users

- Primary: the author, running the app on their own laptop.
- Later: classmates, each running their own copy on their own machine with their own folder and index.
- No shared server, no accounts, no authentication.

## Goals

- Ask a question in a localhost browser chat and get an answer grounded in local lecture slides.
- Every slide-grounded answer cites deck name and slide page.
- Follow-up questions that refer to earlier turns work.
- When the slides do not cover the question, answer from the web and label it clearly as internet-sourced, with links.
- Index a local folder of slide PDFs automatically, without touching the CMS (the author lost CMS access).
- Everything runs locally: Ollama models, local vector store, no cloud API keys.

## Non-goals

- Problem sets, past exams, handwritten notes, textbooks, or video lectures (v1 is slides only).
- CMS scraping or syncing.
- Multi-user hosting, accounts, quotas.
- OCR for scanned PDFs.
- Mobile app.
- Cloud model APIs.

## Anti-goals

- An answer that mixes slide and web content without saying which is which.
- A confident answer with no traceable source.
- A silent failure: if indexing fails, the UI must say so, not return empty answers.

## Constraints

- Host machine: Apple M4, 16 GB RAM, macOS. Ollama 0.15.4 installed.
- Host Python is 3.14.3; the LangChain/LangGraph/RAPTOR dependency set is not reliable there. Use `uv` with Python 3.12.
- Model budget with 16 GB: one 7b generator plus one 3b judge plus an embedding model can coexist.
- Local inference latency matters: each extra model call costs 1-3 s.
- Classroom material may be copyrighted or restricted: nothing leaves the machine except explicit web-search queries.

## Decisions made by the user

- Interface: browser page on localhost, served by FastAPI, no frontend build step.
- Models: local via Ollama, no paid APIs.
- Ingestion: a local folder the student fills by hand; app watches it and indexes. No CMS access.
- Scope: slides only for v1.
- Fallback: web search when slides cannot answer, clearly labelled.
- Retrieval structure: hierarchical indexing following RAPTOR.
- Orchestration: adaptive RAG, with a small-model judge for routing and quality checks. Jev (TypeSafe AI) was considered and rejected: it is a hosted service, so it breaks the local-only promise and sends slide content off the machine.
- Follow-up questions must work.

## Architecture

```
RAG/ folder (PDFs)
  └─ ingest: extract text, chunk, keep page metadata
       └─ RAPTOR: cluster chunks, summarise, build tree levels
            └─ Chroma (persistent, one collection, metadata: source, page, level, node id)

browser chat  ──POST /chat──>  LangGraph adaptive graph
                                 route_scope (3b)
                                 retrieve (collapsed tree, k=8)
                                 grade_relevance (vector score, no model)
                                 generate (7b, citations)
                                 check_hallucination (3b)
                                 check_adequacy (3b)
                                 web_search (DuckDuckGo) + web_generate (7b)
                                 rewrite_query (3b, when history exists)
```

### Ingestion

- Folder: `RAG/` by default, override via `STUDYBUDDY_RAG_DIR`.
- Scan on startup, then poll every 5 s. My call: polling, not an OS watcher; 6-10 files do not need event handling.
- Change detection: a JSON manifest (`data/manifest.json`) holds sha256 per file. If every hash matches the manifest, nothing happens. Any add, remove, or change triggers one full rebuild.
- A file is ingested only after its size and mtime stay unchanged across two consecutive polls, so a half-copied PDF is not indexed. A file that fails extraction three times is quarantined until its hash changes; otherwise it would rebuild on every poll.
- Extraction: `pymupdf4llm`, preserving page numbers as metadata.
- Chunking: `RecursiveCharacterTextSplitter`, 800 characters, 100 overlap. Skip chunks under 40 characters (slide furniture, page numbers).
- A single PDF failing must not abort the whole ingest; log it and report it in `/status`.
- Rebuilds are atomic: build into a new Chroma collection (`studybuddy_v<N>`), then swap the active collection pointer and update the manifest. Chat keeps serving the old index until the swap. A failed rebuild leaves the old index live and reports the error.
- Any change triggers a full RAPTOR rebuild. My call: no incremental tree. This is slow (minutes) and acceptable for 6-10 decks; mark it with a `shortcut:` comment in code.

### RAPTOR tree

- Cluster leaf chunks with UMAP (reduce to ~10 dims) plus Gaussian mixture, then summarise each cluster with the 7b model.
- Repeat one more level over the summaries.
- Store every node in Chroma with `level` and `parent_id` metadata.
- Retrieval: collapsed-tree retrieval (all levels in one index), k=8.

### Adaptive graph

Nodes and edges:

1. `route_scope` (3b judge, structured yes/no): is the question about this course material? Web question goes to `web_search`.
2. `retrieve`: collapsed-tree retrieval, k=8, with scores.
3. `grade_relevance`: deterministic, no model call. Relevant only if at least one chunk exists AND the top cosine similarity ≥ 0.35 (Chroma collection uses cosine space; threshold tuned during eval). Not relevant goes to `web_search`.
4. `generate`: 7b, prompt restricted to retrieved chunks, must cite `deck, page` inline, must say "not in the slides" when unsupported.
5. `check_hallucination` (3b): is every claim supported by the retrieved chunks? Fail means regenerate once with the judge's complaint; a second fail goes to web.
6. `check_adequacy` (3b): does the answer actually answer the question? Fail goes to web.
7. `web_search`: DuckDuckGo search, snippets first, fetch up to 2 pages when snippets are thin. `web_generate` answers from those results and the reply is labelled "From the web" with links.
8. `rewrite_query` (3b): when history exists, rewrite the latest turn into a standalone question before routing and retrieval. Last 6 turns kept. Retrieval always uses the standalone query; history goes to the generator as context only.

Loop caps: one regeneration, one web attempt. No unbounded cycles.

No answer is streamed token by token. The generated answer is buffered, both judges run, and only a passing answer is emitted. A rejected answer is never shown. This keeps the trust promise and keeps the graph acyclic in its visible output: one turn produces exactly one answer.

If both the slide path and web search fail, the reply is: "Your slides do not cover this, and web search failed." plus the search error, as a notice. No unlabelled answer is produced.

Out-of-scope questions go straight to web, so `route_scope` must be conservative: uncertain means treat as in scope and let the relevance and adequacy checks handle it.

### API and UI

- `POST /chat` — body `{message, history, session_id}` where `history` is the last 6 turns as `[{role, content}]` provided by the browser. No server-side session state. Response is newline-delimited JSON events: `{type: "answer", text, source}`, `{type: "sources", items}`, `{type: "notice", text}`, `{type: "error", message}`, `{type: "done"}`. `source` is `"slides"` or `"web"`. `error` is terminal. Only the answer text is delayed; sources arrive with it.
- `GET /status` — indexed decks, chunk count, tree depth, index state (`idle`/`building`), progress, last error per file, model names, active collection.
- `POST /reindex` — starts a rebuild and returns 202 immediately with the current state; the browser polls `/status`.
- Static `index.html` plus vanilla JS: chat log, message input, sources under each answer, a banner for web answers, a status strip, and a "rebuilding" indicator. No build step, no framework.
- History lives in browser session storage and is sent with each request; a reload keeps it, closing the tab clears it.

### Models

- Generation: `qwen2.5:7b-instruct` (fallback `llama3.1:8b`).
- Judge, routing, rewriting: `llama3.2:3b`.
- Embeddings: `nomic-embed-text`.
- All through `langchain-ollama`, `temperature=0` for judge and rewrite calls.

### Evaluation

- No answer key exists yet. The author supplies the PDF decks by placing them in `RAG/`. The builder then drafts ~10 questions from the real decks, with expected answer points, plus one out-of-scope question for the web path and one unanswerable question for refusal. The author corrects the draft.
- The eval ships with whatever decks are present. If `RAG/` is empty, the script exits with a message telling the author to add decks.
- Threshold tuning happens against the eval, not before it.
- Manual scoring, three columns: correct answer, correct citation, correct behaviour (refused or labelled web).
- Eval script runs the questions through the graph and prints a table for manual marking. Not a CI gate.

## Error handling

- Ollama not running or model missing: `/chat` and `/status` report a clear message naming the fix (`ollama serve`, `ollama pull <model>`), and the UI shows it.
- Empty index: chat says no slides indexed and points at the folder; no model call wasted.
- Web search failure: if the slide path produced a passing answer, it is returned with a notice that web search failed. If it did not, the turn ends with the "slides do not cover this and web search failed" reply.
- PDF extraction failure: reported per file in `/status`; quarantined after three attempts.
- Server crash mid-request: the browser shows the missing `done` event as an error.

## Citations

Citations name the deck file and the PDF page number, shown as `deck.pdf p.42`. Beamer exports can offset printed slide numbers, so the PDF page is the ground truth and is the number shown. No deep links in v1.

## Security and privacy

- Binds to `127.0.0.1` only.
- No telemetry.
- Web search is the only outbound network call, and only when the slide path fails.
- No secrets in the repo.

## Left to the builder

Exact prompt wording, the relevance threshold value, chunk sizes after eval, RAPTOR cluster counts, and UI styling. Any of these can change without changing this design.
