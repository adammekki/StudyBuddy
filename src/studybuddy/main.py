import threading
import time

import httpx
import uvicorn

from studybuddy.api import AppState, create_app
from studybuddy.cli import make_graph
from studybuddy.config import Settings, load_settings
from studybuddy.deps import make_deps
from studybuddy.ingest import (
    plan_rebuild,
    snapshot,
    stable_names,
)
from studybuddy.pipeline import _manifest_path, index_status, rebuild

OLLAMA = "http://127.0.0.1:11434"
POLL_SECONDS = 5
REBUILD_LOCK = threading.Lock()


def list_models() -> list[str]:
    try:
        response = httpx.get(f"{OLLAMA}/api/tags", timeout=5)
        response.raise_for_status()
    except httpx.HTTPError as error:
        raise RuntimeError(
            f"Ollama is not reachable at {OLLAMA}. Run: ollama serve"
        ) from error
    return [model["name"] for model in response.json().get("models", [])]


def check_models(required: list[str]) -> None:
    available = {name.split(":")[0] for name in list_models()}
    missing = [name for name in required if name.split(":")[0] not in available]
    if missing:
        pulls = " && ".join(f"ollama pull {name}" for name in missing)
        raise RuntimeError(f"Ollama is missing models. Run: {pulls}")


def watch(settings: Settings, deps, status: dict, stop: threading.Event) -> None:
    previous = snapshot(settings.rag_dir)
    while not stop.wait(POLL_SECONDS):
        current = snapshot(settings.rag_dir)
        stable = stable_names(previous, current)
        previous = current
        if not stable:
            continue
        files, _ = plan_rebuild(settings.rag_dir, _load(settings), stable)
        if not files:
            continue
        _run_rebuild(settings, deps, status, stable)


def _stable_now(settings: Settings) -> set[str]:
    first = snapshot(settings.rag_dir)
    time.sleep(POLL_SECONDS)
    return stable_names(first, snapshot(settings.rag_dir))


def _run_rebuild(settings: Settings, deps, status: dict, names=None):
    if not REBUILD_LOCK.acquire(blocking=False):
        return None
    status["state"] = "building"
    try:
        result = rebuild(settings, deps, names)
        status["last_result"] = result
        status["state"] = "failed" if result["errors"] else "idle"
        return result
    except Exception as error:
        status["last_error"] = str(error)
        status["state"] = "failed"
        return None
    finally:
        REBUILD_LOCK.release()


def _load(settings: Settings) -> dict:
    from studybuddy.ingest import load_manifest

    return load_manifest(_manifest_path(settings))


def build_state(settings: Settings, deps) -> tuple[AppState, dict]:
    graph = make_graph(settings, deps)
    status = {"state": "idle"}

    def run(question: str, history: list[dict]) -> dict:
        if not index_status(settings)["collection"]:
            return {"error": "No slides indexed. Put PDFs in the RAG folder."}
        result = graph.invoke(
            {"question": question, "history": history, "attempts": 0}
        )
        seen = []
        for chunk in result.get("chunks", []):
            if chunk["source"] == "summary":
                continue
            item = {"deck": chunk["source"], "page": chunk["page"]}
            if item not in seen:
                seen.append(item)
        return {
            "answer": result.get("answer", ""),
            "source": result.get("source", "slides"),
            "sources": seen if result.get("source") == "slides" else [],
        }

    def status_view() -> dict:
        view = index_status(settings)
        view["state"] = status["state"]
        view["models"] = {
            "generator": settings.gen_model,
            "judge": settings.judge_model,
            "embeddings": settings.embed_model,
        }
        return view

    def reindex() -> None:
        if status.get("state") == "building":
            return
        threading.Thread(
            target=_reindex_now, args=(settings, deps, status), daemon=True
        ).start()

    return AppState(
        run=run, status=status_view, reindex=reindex, rag_dir=settings.rag_dir
    ), status


def _reindex_now(settings: Settings, deps, status: dict) -> None:
    _run_rebuild(settings, deps, status, _stable_now(settings))


def main() -> None:
    settings = load_settings()
    check_models([settings.gen_model, settings.judge_model, settings.embed_model])
    deps = make_deps(settings)

    if index_status(settings)["collection"] is None:
        print("No index yet; building one from", settings.rag_dir)
        print(rebuild(settings, deps, _stable_now(settings)))

    state, status = build_state(settings, deps)
    stop = threading.Event()
    threading.Thread(
        target=watch, args=(settings, deps, status, stop), daemon=True
    ).start()
    uvicorn.run(create_app(state), host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
