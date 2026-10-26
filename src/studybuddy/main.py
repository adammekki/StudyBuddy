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
    save_manifest,
    snapshot,
    stable_names,
)
from studybuddy.pipeline import _manifest_path, index_status, rebuild

OLLAMA = "http://127.0.0.1:11434"
POLL_SECONDS = 5


def list_models() -> list[str]:
    response = httpx.get(f"{OLLAMA}/api/tags", timeout=5)
    response.raise_for_status()
    return [model["name"] for model in response.json().get("models", [])]


def check_models(required: list[str]) -> None:
    available = set(list_models())
    missing = [name for name in required if name not in available]
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
        files, manifest = plan_rebuild(
            settings.rag_dir, _load(settings), stable
        )
        if not files:
            continue
        status["state"] = "building"
        try:
            result = rebuild(settings, deps, stable)
            status["last_result"] = result
            status["state"] = "failed" if result["errors"] else "idle"
        except Exception as error:
            status["last_error"] = str(error)
            status["state"] = "failed"


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
            "sources": seen,
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
        threading.Thread(
            target=_reindex_now, args=(settings, deps, status), daemon=True
        ).start()

    return AppState(run=run, status=status_view, reindex=reindex), status


def _reindex_now(settings: Settings, deps, status: dict) -> None:
    status["state"] = "building"
    try:
        status["last_result"] = rebuild(settings, deps)
        status["state"] = "failed" if status["last_result"]["errors"] else "idle"
    except Exception as error:
        status["last_error"] = str(error)
        status["state"] = "failed"


def main() -> None:
    settings = load_settings()
    check_models([settings.gen_model, settings.judge_model, settings.embed_model])
    deps = make_deps(settings)

    if index_status(settings)["collection"] is None:
        print("No index yet; building one from", settings.rag_dir)
        print(rebuild(settings, deps))

    state, status = build_state(settings, deps)
    stop = threading.Event()
    threading.Thread(
        target=watch, args=(settings, deps, status, stop), daemon=True
    ).start()
    uvicorn.run(create_app(state), host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
