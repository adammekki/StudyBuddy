# StudyBuddy / MaiaGPT

Local RAG chat over your own lecture slides. Answers cite deck and page; anything from the web is labelled.

Each student runs their own copy: their `RAG/` folder, their own index, their own Ollama.

## Requirements

- macOS or Linux, 16 GB RAM recommended
- [Ollama](https://ollama.com) running locally
- [uv](https://docs.astral.sh/uv/)

```bash
ollama pull gemma4:e4b
ollama pull nomic-embed-text
uv sync
```

## Start the server

```bash
uv run python -m studybuddy.main
```

Then open http://127.0.0.1:8000.

First start builds the index if none exists. It also watches `RAG/` and rebuilds when slides change. Vision captions run once per visual slide and are cached in `data/captions/`.

The server stays in the foreground, logging each request.

## Stop the server

Press `Ctrl-C` in the terminal that runs it.

If it was started in the background and you cannot reach that terminal:

```bash
pkill -f "studybuddy.main"
```

## Everyday commands

```bash
uv run python -m studybuddy.cli status          # index state, decks, chunk count
uv run python -m studybuddy.cli index           # force an index build
uv run python -m studybuddy.cli ask "question"  # one question in the terminal
uv run pytest -q                                # test suite
```

## Trace the graph with LangSmith

LangSmith is hosted. Turning it on uploads each question, the slide excerpts and captions retrieved for it, the prompts sent to the model, and the answers, to LangChain's servers. Do not enable it on material you cannot share.

```bash
cp .env.example .env
# paste your key from https://smith.langchain.com into LANGSMITH_API_KEY
uv run python -m studybuddy.main
```

Then open your project at smith.langchain.com. Every chat turn appears as a trace: `rewrite → step_back → route → retrieve → generate → judge_hallucination → judge_adequacy`, with inputs, outputs, retrieved chunks and timings per node.

`.env` is read automatically and is git-ignored. To keep the trace shape without the content, set `LANGSMITH_HIDE_INPUTS=true` and `LANGSMITH_HIDE_OUTPUTS=true`. Unset `LANGSMITH_TRACING` to turn tracing off.

## Layout

- `RAG/` — put your slide PDFs here (git-ignored)
- `data/` — Chroma index, manifest, caption cache (git-ignored)
- `src/studybuddy/` — pipeline, graph, API, static UI
- `docs/superpowers/` — design and implementation plan

## How it answers

1. Rewrite follow-up questions into a standalone question, then step back to a more general one.
2. Route: in-course questions retrieve from the RAPTOR tree (chunks + cluster summaries); out-of-course questions go to web search.
3. Generate from the retrieved excerpts with `[deck.pdf p.N]` citations.
4. A judge checks that every claim is supported and that the answer addresses the question. A failure regenerates once, then falls back to the web, labelled "From the web".

Model: `gemma4:e4b` for generation, judging and vision captions, `nomic-embed-text` for embeddings. Everything runs locally.
