from pathlib import Path

from studybuddy.chunking import extract_pages, make_chunks
from studybuddy.config import Settings
from studybuddy.ingest import (
    load_manifest,
    plan_rebuild,
    record_failure,
    save_manifest,
    sha256_file,
)
from studybuddy.raptor import build_tree
from studybuddy.store import Store


class NoSlidesError(Exception):
    """The folder has no indexable slides."""


def _manifest_path(settings: Settings) -> Path:
    return settings.data_dir / "manifest.json"


def rebuild(settings: Settings, deps, names: set[str] | None = None) -> dict:
    manifest = load_manifest(_manifest_path(settings))
    files, manifest = plan_rebuild(settings.rag_dir, manifest, names)
    errors: list[dict] = []

    if not files:
        quarantined = [
            {"file": name, "error": entry.get("error")}
            for name, entry in manifest.get("files", {}).items()
            if entry.get("quarantined")
        ]
        save_manifest(_manifest_path(settings), manifest)
        if quarantined:
            errors.extend(quarantined)
        return {
            "collection": manifest.get("collection"),
            "decks": 0,
            "chunks": 0,
            "levels": 0,
            "errors": errors,
        }

    chunks: list[dict] = []
    decks = 0
    for path in files:
        try:
            pages = extract_pages(path)
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
    store.activate(collection)

    manifest["collection"] = collection
    manifest["files"] = {
        name: {**entry, "sha": sha256_file(settings.rag_dir / name)}
        for name, entry in manifest.get("files", {}).items()
    }
    save_manifest(_manifest_path(settings), manifest)

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
