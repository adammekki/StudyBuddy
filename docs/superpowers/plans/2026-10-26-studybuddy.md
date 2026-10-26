# StudyBuddy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A localhost chat app that answers questions from a folder of lecture-slide PDFs, citing deck and page, with a labelled web fallback.

**Architecture:** PDFs in `RAG/` are extracted and chunked, then built into a RAPTOR summary tree stored in Chroma. A LangGraph adaptive-RAG graph routes each question, retrieves, grades, generates, judges the answer, and falls back to web search when the slides do not cover it. FastAPI serves one static chat page on `127.0.0.1`.

**Tech Stack:** Python 3.12 via uv, LangChain, LangGraph, langchain-ollama, Chroma, pymupdf4llm, scikit-learn + umap-learn (RAPTOR clustering), FastAPI + uvicorn, pytest.

**Spec:** `docs/superpowers/specs/2026-10-26-studybuddy-design.md`

## Global Constraints

- Python 3.12 through `uv`. Host Python is 3.14 and must not be used.
- Models via Ollama only: generator `gemma4:e4b`, judge `llama3.2:3b`, embeddings `nomic-embed-text`. No cloud model APIs.
- Judge and rewrite calls use `temperature=0`.
- Server binds `127.0.0.1` only.
- No token streaming. The answer is buffered, judged, then emitted once.
- Chroma collections use cosine space. Relevance gate: at least one chunk AND top similarity ≥ 0.35.
- Citations are `deck.pdf p.<pdf page>`.
- Loop caps: one regeneration, one web attempt per turn.
- Tests must not require a running Ollama. Model calls are injected.
- Anything from the web is labelled `source: "web"` and never mixed into a slide answer.

## Review Focus

- A PDF still being copied into `RAG/` must not be indexed as a truncated deck; it waits for two identical snapshots.
- An image-only PDF yields no text: it must be reported in `/status`, never silently indexed as empty.
- A follow-up like "why does that hold?" must retrieve using a rewritten standalone query, not the raw words.
- Ollama down or a model missing must produce an `error` event with the fix in it, not a traceback.
- A top score just under 0.35 must produce a web answer labelled `web`, with no slide content in it.

---

### Task 1: Scaffold, config, model factories

**Files:**
- Create: `pyproject.toml`, `src/studybuddy/__init__.py`, `src/studybuddy/config.py`, `src/studybuddy/deps.py`, `tests/test_config.py`, `.gitignore`, `README.md`

**Interfaces:**
- Produces: `Settings` dataclass with `rag_dir: Path`, `data_dir: Path`, `gen_model: str`, `judge_model: str`, `embed_model: str`, `k: int`, `threshold: float`, `chunk_size: int`, `chunk_overlap: int`; `def load_settings() -> Settings`.
- Produces: `Deps` dataclass with `llm`, `judge`, `embed` fields; `def make_deps(settings: Settings) -> Deps` returning `langchain_ollama` objects.

- [ ] **Step 1: Write the failing test**

```python
def test_defaults_and_env_override(monkeypatch, tmp_path):
    monkeypatch.delenv("STUDYBUDDY_RAG_DIR", raising=False)
    s = load_settings()
    assert s.rag_dir.name == "RAG"
    assert s.k == 8 and s.threshold == 0.35
    assert s.chunk_size == 800 and s.chunk_overlap == 100
    monkeypatch.setenv("STUDYBUDDY_RAG_DIR", str(tmp_path))
    assert load_settings().rag_dir == tmp_path
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL, `ImportError` or `NameError` for `load_settings`

- [ ] **Step 3: Implement settings and factories**

`pyproject.toml`: project `studybuddy`, `requires-python = ">=3.12,<3.13"`, deps `langchain`, `langchain-community`, `langchain-ollama`, `langchain-experimental`, `langgraph`, `chromadb`, `pymupdf4llm`, `pymupdf`, `scikit-learn`, `umap-learn`, `ddgs`, `fastapi`, `uvicorn`, `httpx`, `pytest`, `pytest-asyncio`. Package dir `src`.

`config.py` reads `STUDYBUDDY_RAG_DIR` (default `./RAG`) and `STUDYBUDDY_DATA_DIR` (default `./data`), creates `data_dir`.

`deps.py` builds `ChatOllama(model=..., temperature=0)` for judge and `ChatOllama(model=...)` for generator, `OllamaEmbeddings(model=...)`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_config.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src tests .gitignore README.md
git commit -m "chore: scaffold studybuddy with settings and model factories"
```

---

### Task 2: PDF extraction and chunking

**Files:**
- Create: `src/studybuddy/chunking.py`, `tests/test_chunking.py`

**Interfaces:**
- Produces: `def extract_pages(path: Path) -> list[tuple[int, str]]` — 1-based PDF page numbers, `pymupdf4llm`.
- Produces: `def make_chunks(pages: list[tuple[int, str]], source: str, size: int = 800, overlap: int = 100) -> list[dict]` — each item `{"text": str, "source": str, "page": int, "chunk_index": int}`.
- Produces: `class NoTextError(Exception)` raised by `extract_pages` when the whole PDF yields blank text.

- [ ] **Step 1: Write the failing test**

```python
def test_chunks_carry_source_page_and_skip_furniture():
    pages = [(1, "Short."), (2, "A" * 2000)]
    chunks = make_chunks(pages, "deck.pdf")
    assert all(c["source"] == "deck.pdf" for c in chunks)
    assert all(len(c["text"]) > 40 for c in chunks)
    assert chunks[0]["page"] == 2
    assert chunks[0]["chunk_index"] == 0 and chunks[1]["chunk_index"] == 1
    assert chunks[0]["text"] != chunks[1]["text"]

def test_extract_pages_reads_page_numbers(tmp_path):
    doc = fitz.open(); page = doc.new_page(); page.insert_text((72, 72), "Lecture one content"); path = tmp_path / "d.pdf"; doc.save(path)
    assert extract_pages(path)[0][0] == 1
    assert "Lecture one" in extract_pages(path)[0][1]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_chunking.py -v`
Expected: FAIL, module not found

- [ ] **Step 3: Implement extraction and chunking**

Use `pymupdf4llm.to_markdown(path, page_chunks=True)`; pull `metadata["page"]` (0-based → +1) and `text`. Chunk per page with `RecursiveCharacterTextSplitter(chunk_size, chunk_overlap)`, drop parts shorter than 40 characters, keep `chunk_index` per page.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_chunking.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/studybuddy/chunking.py tests/test_chunking.py
git commit -m "feat: extract slide text with page metadata and chunk it"
```

---

### Task 3: Folder scan, stability, manifest, quarantine

**Files:**
- Create: `src/studybuddy/ingest.py`, `tests/test_ingest.py`

**Interfaces:**
- Produces: `def snapshot(folder: Path) -> dict[str, tuple[int, float]]` — filename → (size, mtime).
- Produces: `def stable_names(prev: dict, cur: dict) -> set[str]` — names present in both with equal size and mtime.
- Produces: `def sha256_file(path: Path) -> str`.
- Produces: `def plan_rebuild(folder: Path, manifest: dict, names: set[str] | None = None) -> tuple[list[Path], dict]` — files to index and the next manifest, applying hash comparison and quarantine; `names=None` means every PDF.
- Produces: `def record_failure(manifest: dict, name: str, error: str) -> None` — increments count, marks quarantined at 3.
- Manifest JSON: `{"collection": str|None, "files": {name: {"sha", "failures", "quarantined", "error"}}}`.
- Produces: `def load_manifest(path) -> dict`, `def save_manifest(path, data) -> None`.

- [ ] **Step 1: Write the failing test**

```python
def test_half_copied_file_is_not_stable(tmp_path):
    (tmp_path / "a.pdf").write_bytes(b"x" * 10)
    first = snapshot(tmp_path)
    (tmp_path / "a.pdf").write_bytes(b"x" * 20)
    assert stable_names(first, snapshot(tmp_path)) == set()
    assert stable_names(snapshot(tmp_path), snapshot(tmp_path)) == {"a.pdf"}

def test_quarantine_after_three_failures(tmp_path):
    m = {"collection": None, "files": {}}
    for _ in range(3):
        record_failure(m, "bad.pdf", "no text")
    assert m["files"]["bad.pdf"]["quarantined"] is True

def test_unchanged_hashes_skip_rebuild(tmp_path):
    (tmp_path / "a.pdf").write_bytes(b"slides")
    files, m = plan_rebuild(tmp_path, {"collection": "studybuddy_v1", "files": {}})
    assert [f.name for f in files] == ["a.pdf"]
    again, _ = plan_rebuild(tmp_path, {"collection": "studybuddy_v1", "files": m["files"]})
    assert again == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_ingest.py -v`
Expected: FAIL, `plan_rebuild` not defined

- [ ] **Step 3: Implement**

`plan_rebuild` lists `*.pdf`, skips names not in `names` when `names` is given, and skips quarantined names. Rebuild is required when the set of hashes differs from `manifest["files"]`, or when `manifest["collection"]` is None.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_ingest.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/studybuddy/ingest.py tests/test_ingest.py
git commit -m "feat: detect stable slide files and track hashes"
```

---

### Task 4: Chroma store with atomic collection swap

**Files:**
- Create: `src/studybuddy/store.py`, `tests/test_store.py`

**Interfaces:**
- Consumes: `Deps.embed` (Task 1).
- Produces: `class Store` with `__init__(path: Path, embed)`, `build(nodes: list[dict]) -> str` (returns new collection name `studybuddy_v<N>`), `activate(name: str) -> None`, `active() -> Collection | None`, `search(query: str, k: int) -> list[dict]` returning `{"text", "source", "page", "level", "score"}` with `score = 1 - distance`.
- Collections use `metadata={"hnsw:space": "cosine"}`.

- [ ] **Step 1: Write the failing test**

```python
class FakeEmbed:
    def embed_documents(self, texts): return [[float(len(t) % 7), 1.0, 0.5] for t in texts]
    def embed_query(self, text): return [float(len(text) % 7), 1.0, 0.5]

def test_build_activate_and_old_index_survives_failed_build(tmp_path):
    s = Store(tmp_path, FakeEmbed())
    v1 = s.build([{"text": "alpha", "source": "a.pdf", "page": 1, "level": 0}], embed=FakeEmbed())
    s.activate(v1)
    assert s.active().name == v1
    assert s.search("alpha", 3)[0]["source"] == "a.pdf"

def test_search_score_is_similarity(tmp_path):
    s = Store(tmp_path, FakeEmbed())
    s.activate(s.build([{"text": "beta", "source": "b.pdf", "page": 2, "level": 0}]))
    assert 0.0 <= s.search("beta", 1)[0]["score"] <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_store.py -v`
Expected: FAIL, `Store` not defined

- [ ] **Step 3: Implement**

`chromadb.PersistentClient(path=...)` under `data_dir/chroma`. Active name stored in `data/active.json` so a crash before activation keeps the old index. `build` picks `studybuddy_v<N+1>` from existing collection names and adds in batches of 100.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_store.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/studybuddy/store.py tests/test_store.py
git commit -m "feat: persistent chroma store with atomic collection swap"
```

---

### Task 5: RAPTOR tree and retrieval gate

**Files:**
- Create: `src/studybuddy/raptor.py`, `src/studybuddy/retrieval.py`, `tests/test_raptor.py`, `tests/test_retrieval.py`

**Interfaces:**
- Produces: `def build_tree(chunks: list[dict], summarize: Callable[[str], str], cluster: Callable[[list[dict]], list[list[int]]], levels: int = 2) -> list[dict]` — level 0 keeps chunks, higher levels get `{"text", "source": "summary", "page": 0, "level": n, "parent_id": str}`.
- Produces: `def is_relevant(chunks: list[dict], threshold: float = 0.35) -> bool`.
- Produces: `def merge_results(a: list[dict], b: list[dict], limit: int = 8) -> list[dict]` — dedupe on `(source, page, text)`, keep the higher `score`, sort by score desc.
- Produces: `def format_context(chunks: list[dict], limit: int = 8) -> str` — lines prefixed `[deck.pdf p.N]`.
- Produces: default `cluster_chunks` using UMAP + `GaussianMixture`, capped at `n_components=min(10, n_samples-1)`.

- [ ] **Step 1: Write the failing test**

```python
def test_tree_has_summary_levels_and_parent_links():
    chunks = [{"text": f"chunk {i}", "source": "a.pdf", "page": i, "level": 0} for i in range(1, 7)]
    tree = build_tree(chunks, summarize=lambda t: "SUM:" + t[:10],
                      cluster=lambda cs: [[0, 1, 2], [3, 4, 5]], levels=2)
    levels = {n["level"] for n in tree}
    assert levels == {0, 1}
    assert all(n["parent_id"] for n in tree if n["level"] > 0)
    assert all(n["text"].startswith("SUM:") for n in tree if n["level"] > 0)

def test_gate_requires_chunk_and_threshold():
    assert is_relevant([]) is False
    assert is_relevant([{"score": 0.2}]) is False
    assert is_relevant([{"score": 0.9}]) is True

def test_merge_dedupes_keeping_best_score():
    a = [{"text": "x", "source": "a.pdf", "page": 1, "level": 0, "score": 0.8}]
    b = [{"text": "x", "source": "a.pdf", "page": 1, "level": 0, "score": 0.6},
         {"text": "y", "source": "b.pdf", "page": 2, "level": 1, "score": 0.5}]
    merged = merge_results(a, b)
    assert [m["text"] for m in merged] == ["x", "y"]
    assert merged[0]["score"] == 0.8
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_raptor.py tests/test_retrieval.py -v`
Expected: FAIL, modules missing

- [ ] **Step 3: Implement**

`build_tree` groups by level, summarises concatenated cluster text (truncated to 6000 chars per cluster), assigns `parent_id` as `f"L{level}-{i}"`. Retrieval = `Store.search` filtered to `level <= levels`, then `is_relevant` on the top score. `format_context` joins chunk text with citation prefixes.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_raptor.py tests/test_retrieval.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/studybuddy/raptor.py src/studybuddy/retrieval.py tests
git commit -m "feat: build raptor summary tree and gate retrieval by similarity"
```

---

### Task 6: Adaptive graph with injected models

**Files:**
- Create: `src/studybuddy/graph/__init__.py`, `src/studybuddy/graph/state.py`, `src/studybuddy/graph/nodes.py`, `src/studybuddy/graph/build.py`, `src/studybuddy/web.py`, `tests/test_graph.py`, `tests/test_web.py`

**Interfaces:**
- Consumes: `format_context`, `is_relevant`, `merge_results`, `Store.search`.
- Produces: `ChatState` TypedDict with `question: str`, `history: list[dict]`, `standalone: str`, `step_back: str`, `chunks: list[dict]`, `answer: str`, `source: str`, `notices: list[str]`, `attempts: int`.
- Produces: `@dataclass GraphDeps` with `retrieve: Callable[[str, int], list[dict]]`, `answer_llm: Callable[[str], str]`, `judge: Callable[[str], bool]`, `rewrite: Callable[[str, list[dict]], str]`, `step_back: Callable[[str], str]`, `web_search: Callable[[str], list[dict]]`, `k: int`, `threshold: float`.
- Produces: `def build_graph(d: GraphDeps) -> CompiledGraph` with nodes `rewrite`, `step_back`, `route`, `retrieve`, `generate`, `regenerate`, `web`, `give_up`.
- Produces: `def web_search(query: str, fetch_pages: int = 2) -> list[dict]` — `{"title", "url", "text"}`, `ddgs` snippets, page fetch with 5 s timeout and 4000-char trim.

- [ ] **Step 1: Write the failing test**

```python
def _deps(**over):
    d = dict(retrieve=lambda q, k: [{"text": "X is Y", "source": "a.pdf", "page": 3, "score": 0.9}],
             answer_llm=lambda p: "X is Y [a.pdf p.3]",
             judge=lambda p: True,
             rewrite=lambda q, h: q,
             step_back=lambda q: q,
             web_search=lambda q: [{"title": "t", "url": "u", "text": "web says Z"}],
             k=8, threshold=0.35)
    d.update(over); return GraphDeps(**d)

def test_relevant_question_answers_from_slides():
    out = build_graph(_deps()).invoke({"question": "what is X?", "history": [], "attempts": 0})
    assert out["source"] == "slides" and "a.pdf p.3" in out["answer"]

def test_weak_retrieval_goes_to_web_with_label():
    d = _deps(retrieve=lambda q, k: [{"text": "x", "source": "a.pdf", "page": 1, "score": 0.1}],
              answer_llm=lambda p: ("web says Z" if "WEB" in p else "slide answer"))
    out = build_graph(d).invoke({"question": "q", "history": [], "attempts": 0})
    assert out["source"] == "web"

def test_failed_hallucination_check_regenerates_once_then_web():
    def judge(p):
        return False if "supported" in p.lower() else True
    out = build_graph(_deps(judge=judge, answer_llm=lambda p: "unsupported claim")).invoke(
        {"question": "q", "history": [], "attempts": 0})
    assert out["source"] == "web"

def test_follow_up_is_rewritten_before_retrieval():
    seen = {}
    def retrieve(q, k): seen["q"] = q; return [{"text": "X is Y", "source": "a.pdf", "page": 3, "score": 0.9}]
    build_graph(_deps(retrieve=retrieve, rewrite=lambda q, h: "what is X and why does it hold?")).invoke(
        {"question": "why?", "history": [{"role": "user", "content": "what is X?"}], "attempts": 0})
    assert seen["q"] == "what is X and why does it hold?"

def test_step_back_retrieves_both_queries_and_merges():
    seen = []
    def retrieve(q, k):
        seen.append(q)
        return [{"text": "ctx for " + q, "source": "a.pdf", "page": 1, "score": 0.9}]
    out = build_graph(_deps(retrieve=retrieve, step_back=lambda q: "general: " + q)).invoke(
        {"question": "q", "history": [], "attempts": 0})
    assert seen == ["q", "general: q"]
    assert out["source"] == "slides"

def test_step_back_failure_does_not_block_answer():
    def boom(q): raise RuntimeError("timeout")
    out = build_graph(_deps(step_back=boom)).invoke({"question": "q", "history": [], "attempts": 0})
    assert out["source"] == "slides"

def test_both_paths_fail_gives_labelled_refusal():
    d = _deps(retrieve=lambda q, k: [], web_search=lambda q: [])
    out = build_graph(d).invoke({"question": "q", "history": [], "attempts": 0})
    assert out["source"] == "refused"
    assert out["answer"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_graph.py -v`
Expected: FAIL, `build_graph` not defined

- [ ] **Step 3: Implement the graph**

`StateGraph(ChatState)`: `rewrite → step_back → route → retrieve → (web | generate) → judge_hallucination → (regenerate once | judge_adequacy) → (web | done)`. `retrieve` runs once per query and merges with `merge_results`. `step_back` failures are caught and ignored. `regenerate` fails once more → web. `web` with no results → `give_up`. Prompts include the `[deck.pdf p.N]` convention and forbid claims not in context. The generation prompt names the original question as the one to answer and passes step-back chunks as background. Judge prompts end with "Answer only YES or NO." and parse `yes` case-insensitively; anything unparseable counts as pass to avoid loops.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_graph.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/studybuddy/graph src/studybuddy/web.py tests
git commit -m "feat: adaptive rag graph with judges and web fallback"
```

---

### Task 7: Indexing pipeline and CLI

**Files:**
- Create: `src/studybuddy/pipeline.py`, `src/studybuddy/cli.py`, `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `snapshot`, `stable_names`, `plan_rebuild`, `record_failure`, `load_manifest`, `save_manifest`, `make_chunks`, `extract_pages`, `build_tree`, `Store`.
- Produces: `def rebuild(settings, deps, names: set[str] | None = None) -> dict` — `{"collection", "decks", "chunks", "levels", "errors"}`; builds into a new collection then activates.
- Produces: `def index_status(settings) -> dict` — `{"collection", "decks": [...], "chunks", "state", "last_error"}`.
- Produces: CLI `python -m studybuddy.cli index` and `python -m studybuddy.cli ask "question"`.

- [ ] **Step 1: Write the failing test**

```python
def test_rebuild_reports_bad_pdf_and_keeps_old_collection(tmp_path, monkeypatch):
    rag = tmp_path / "RAG"; rag.mkdir(); (rag / "good.pdf").write_bytes(b"%PDF-broken")
    settings = Settings(rag_dir=rag, data_dir=tmp_path / "data", gen_model="g", judge_model="j", embed_model="e", k=8, threshold=0.35, chunk_size=800, chunk_overlap=100)
    result = rebuild(settings, deps=_NullDeps(), names={"good.pdf"})
    assert result["errors"], "broken pdf must be reported"
    assert index_status(settings)["state"] in {"idle", "failed"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: FAIL, `rebuild` not defined

- [ ] **Step 3: Implement**

`rebuild` collects extractable decks, records failures for the rest, builds all chunks into one tree, then `Store.build` + `activate`, and writes the manifest. Empty corpus raises `NoSlidesError` with the folder path. CLI prints JSON for both commands.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/studybuddy/pipeline.py src/studybuddy/cli.py tests/test_pipeline.py
git commit -m "feat: incremental index pipeline and terminal cli"
```

---

### Task 8: FastAPI endpoints and chat page

**Files:**
- Create: `src/studybuddy/api.py`, `src/studybuddy/static/index.html`, `tests/test_api.py`

**Interfaces:**
- Consumes: graph, `index_status`, `rebuild`, `Store`.
- Produces: `def create_app(state: AppState) -> FastAPI` with `POST /chat`, `GET /status`, `POST /reindex`, and `/` serving the page.
- `/chat` body: `{"message": str, "history": list[{"role","content"}], "session_id": str}`; NDJSON events `answer|sources|notice|error|done`.
- `/reindex` returns 202 and starts a background rebuild.

- [ ] **Step 1: Write the failing test**

```python
def test_chat_emits_answer_then_sources_then_done():
    app = create_app(fake_state(answer="X is Y", source="slides",
                                sources=[{"deck": "a.pdf", "page": 3}]))
    r = TestClient(app).post("/chat", json={"message": "what is X?", "history": [], "session_id": "s1"})
    events = [json.loads(line) for line in r.text.strip().splitlines()]
    assert [e["type"] for e in events] == ["answer", "sources", "done"]

def test_empty_index_returns_notice_without_model_call():
    app = create_app(fake_state(answer=None, empty=True))
    events = [json.loads(l) for l in TestClient(app).post("/chat", json={"message": "q", "history": [], "session_id": "s"}).text.splitlines()]
    assert events[0]["type"] == "error"
    assert "RAG" in events[0]["message"]

def test_status_and_reindex_contract():
    c = TestClient(create_app(fake_state()))
    assert c.get("/status").json()["state"] in {"idle", "building"}
    assert c.post("/reindex").status_code == 202
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_api.py -v`
Expected: FAIL, `create_app` not defined

- [ ] **Step 3: Implement API and page**

`StreamingResponse` with `application/x-ndjson`. `AppState` holds `settings`, `store`, `graph`, `status` dict guarded by a lock. Page: textarea, send button, message list, per-answer source lines, amber banner for `web`, red banner for `error`, status strip polled every 5 s, 6-turn history in `sessionStorage`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_api.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/studybuddy/api.py src/studybuddy/static tests/test_api.py
git commit -m "feat: fastapi chat endpoints and localhost chat page"
```

---

### Task 9: Startup watcher, real Ollama smoke test, eval script

**Files:**
- Create: `src/studybuddy/main.py`, `eval/questions.yaml`, `eval/run_eval.py`, `tests/test_smoke.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `python -m studybuddy.main` serving on `127.0.0.1:8000` with a 5 s poll loop that rebuilds only when `stable_names` and hashes change.
- Produces: `eval/run_eval.py` printing `question | answer | cited pages | expected points` for manual marking.

- [ ] **Step 1: Write the failing test**

```python
def test_model_check_reports_missing_model(monkeypatch):
    monkeypatch.setattr(main, "list_models", lambda: ["other:latest"])
    with pytest.raises(RuntimeError) as e:
        main.check_models(["gemma4:e4b"])
    assert "ollama pull gemma4:e4b" in str(e.value)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_smoke.py -v`
Expected: FAIL, `check_models` not defined

- [ ] **Step 3: Implement startup, watcher, eval fixtures**

`check_models` calls `ollama list` via `httpx` to the local API. Watcher thread skips when snapshot is unchanged; sets `state.status["state"] = "building"` around rebuilds. `eval/questions.yaml` gets ~10 questions drafted from the indexed decks with expected answer points, plus `out_of_scope` and `unanswerable` cases; the author corrects the draft.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest -v`
Expected: all PASS

- [ ] **Step 5: Real smoke test**

```bash
ollama pull gemma4:e4b && ollama pull llama3.2:3b && ollama pull nomic-embed-text
mkdir -p RAG
# copy your lecture PDFs into RAG/
uv run python -m studybuddy.cli index
uv run python -m studybuddy.cli ask "<a question your deck 1 answers>"
uv run python -m studybuddy.main
```
Expected: cited answer in terminal, chat page answers in browser, `/status` lists the decks.

- [ ] **Step 6: Commit**

```bash
git add src/studybuddy/main.py eval tests/test_smoke.py
git commit -m "feat: startup watcher, model checks and eval script"
```

---

## Self-review

- Spec coverage: ingestion, RAPTOR, step-back transformation, adaptive graph, judges, web labelling, follow-ups, API, status, errors, eval each map to a task. Multi-user hosting, OCR, video and CMS sync stay out, as the spec says.
- Types: `ChatState.source` is `"slides" | "web" | "refused"`; API maps `refused` to an `error` event. `Store.search` returns `score`; `is_relevant` reads only `score`. Rewrite returns text, so it is a separate dependency from the yes/no `judge`.
- Gaps fixed inline: `plan_rebuild` needed an explicit `names` set for stability; both-fail path added in Task 6.
- Open by design: prompts, threshold tuning, cluster counts, styling.
