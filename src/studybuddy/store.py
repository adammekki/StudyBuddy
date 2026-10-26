import json
import os
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import EmbeddingFunction

PREFIX = "studybuddy_v"


class _ChromaEmbed(EmbeddingFunction):
    def __init__(self, embed) -> None:
        self._embed = embed

    def name(self) -> str:
        return "studybuddy"

    def __call__(self, input: list[str]) -> list[list[float]]:
        return self._embed.embed_documents(list(input))


class Store:
    def __init__(self, path: Path, embed) -> None:
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.embed = embed
        self.client = chromadb.PersistentClient(path=str(self.path))
        self._active_file = self.path / "active.json"

    def build(self, nodes: list[dict]) -> str:
        existing = [
            c.name for c in self.client.list_collections() if c.name.startswith(PREFIX)
        ]
        number = max([int(n.rsplit("_v", 1)[1]) for n in existing], default=0) + 1
        name = f"{PREFIX}{number}"
        collection = self.client.create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=_ChromaEmbed(self.embed),
        )
        for start in range(0, len(nodes), 100):
            batch = nodes[start : start + 100]
            collection.add(
                ids=[f"{start + i}" for i in range(len(batch))],
                documents=[n["text"] for n in batch],
                metadatas=[
                    {
                        "source": n.get("source", ""),
                        "page": int(n.get("page", 0)),
                        "level": int(n.get("level", 0)),
                    }
                    for n in batch
                ],
            )
        return name

    def _write_active(self, name: str) -> None:
        tmp = self._active_file.with_suffix(".tmp")
        tmp.write_text(json.dumps({"collection": name}))
        os.replace(tmp, self._active_file)

    def activate(self, name: str) -> None:
        self._write_active(name)

    def active_name(self) -> str | None:
        if not self._active_file.exists():
            return None
        return json.loads(self._active_file.read_text()).get("collection")

    def active(self):
        name = self.active_name()
        if name is None:
            return None
        try:
            return self.client.get_collection(
                name=name, embedding_function=_ChromaEmbed(self.embed)
            )
        except Exception:
            return None

    def search(self, query: str, k: int) -> list[dict]:
        collection = self.active()
        if collection is None:
            return []
        result = collection.query(
            query_embeddings=[self.embed.embed_query(query)],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        for text, meta, distance in zip(
            result["documents"][0], result["metadatas"][0], result["distances"][0]
        ):
            hits.append(
                {
                    "text": text,
                    "source": meta.get("source", ""),
                    "page": meta.get("page", 0),
                    "level": meta.get("level", 0),
                    "score": max(0.0, 1.0 - float(distance)),
                }
            )
        return hits
