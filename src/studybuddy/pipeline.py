import fcntl
from contextlib import contextmanager
from pathlib import Path

from studybuddy.captions import caption_pages
from studybuddy.chunking import extract_pages, make_chunks
from studybuddy.config import Settings
from studybuddy.ingest import (
    load_manifest,
    plan_rebuild,
    record_failure,
    save_manifest,
)
from studybuddy.raptor import build_tree
from studybuddy.store import Store


class NoSlidesError(Exception):
    """The folder has no indexable slides."""


def _manifest_path(settings: Settings) -> Path:
    return settings.data_dir / "manifest.json"


def rebuild(settings: Settings, deps, names: set[str] | None = None) -> dict:
    with build_lock(settings):
        return _rebuild(settings, deps, names)


@contextmanager
def build_lock(settings: Settings):
    """Cross-process lock: the server watcher and a terminal `index` must not build at once."""
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    handle = open(settings.data_dir / "rebuild.lock", "w")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(handle, fcntl.LOCK_UN)
        handle.close()


def _rebuild(settings: Settings, deps, names: set[str] | None = None) -> dict:
    manifest = load_manifest(_manifest_path(settings))
    trigger, manifest = plan_rebuild(settings.rag_dir, manifest, names)
    indexable = [
        path
        for path in sorted(settings.rag_dir.glob("*.pdf"))
        if not manifest.get("files", {}).get(path.name, {}).get("quarantined")
    ]
    errors: list[dict] = []

    if not trigger or not indexable:
        quarantined = [
            {"file": name, "error": entry.get("error")}
            for name, entry in manifest.get("files", {}).items()
            if entry.get("quarantined")
        ]
        errors.extend(quarantined)
        save_manifest(_manifest_path(settings), manifest)
        return {
            "collection": manifest.get("collection"),
            "decks": len(indexable),
            "chunks": 0,
            "levels": 0,
            "errors": errors,
        }

    chunks: list[dict] = []
    decks = 0
    for path in indexable:
        try:
            pages = extract_pages(path)
            captioner = getattr(deps, "caption", None)
            if captioner:
                sha = manifest.get("files", {}).get(path.name, {}).get("sha", path.name)
                captions = caption_pages(
                    path, sha, settings.data_dir / "captions", captioner
                )
                pages = [
                    (
                        number,
                        text
                        + (
                            f"\n\n[Visuals: {captions[number]}]"
                            if captions.get(number)
                            else ""
                        ),
                    )
                    for number, text in pages
                ]
            file_chunks = make_chunks(
                pages,
                path.name,
                size=settings.chunk_size,
                overlap=settings.chunk_overlap,
            )
            if not file_chunks:
                raise ValueError("no usable text")
            chunks.extend(file_chunks)
            decks += 1
            entry = manifest["files"].setdefault(path.name, {})
            entry["failures"] = 0
            entry["error"] = None
            entry["quarantined"] = False
        except Exception as error:  # one bad deck must not stop the rebuild
            record_failure(manifest, path.name, str(error))
            errors.append({"file": path.name, "error": str(error)})

    if not chunks:
        save_manifest(_manifest_path(settings), manifest)
        return {
            "collection": manifest.get("collection"),
            "decks": 0,
            "chunks": 0,
            "levels": 0,
            "errors": errors,
        }

    tree = build_tree(chunks, summarize=lambda text: _summarize(deps, text))
    store = Store(settings.data_dir / "chroma", deps.embed)
    collection = store.build(tree)

    manifest["collection"] = collection
    save_manifest(_manifest_path(settings), manifest)
    store.activate(collection)

    return {
        "collection": collection,
        "decks": decks,
        "chunks": len(chunks),
        "levels": len({node["level"] for node in tree}),
        "errors": errors,
    }


def _summarize(deps, text: str) -> str:
    prompt = (
        "Summarise these lecture slides in a short paragraph that keeps the key "
        f"definitions and results:\n\n{text}"
    )
    return deps.llm.invoke(prompt).content


def index_status(settings: Settings) -> dict:
    manifest = load_manifest(_manifest_path(settings))
    errors = [
        {"file": name, "error": entry.get("error")}
        for name, entry in manifest.get("files", {}).items()
        if entry.get("error")
    ]
    decks = [
        name
        for name, entry in manifest.get("files", {}).items()
        if not entry.get("quarantined")
    ]
    return {
        "collection": manifest.get("collection"),
        "decks": decks,
        "chunks": _chunk_count(settings),
        "state": "failed" if errors else "idle",
        "errors": errors,
    }


def _chunk_count(settings: Settings) -> int:
    store = Store(settings.data_dir / "chroma", _NoEmbed())
    collection = store.active()
    return collection.count() if collection else 0


class _NoEmbed:
    def embed_documents(self, texts):
        return [[0.0] for _ in texts]

    def embed_query(self, text):
        return [0.0]
