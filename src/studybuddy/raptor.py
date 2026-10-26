from collections.abc import Callable

SUMMARY_CHAR_LIMIT = 6000


def build_tree(
    chunks: list[dict],
    summarize: Callable[[str], str],
    cluster: Callable[[list[dict]], list[list[int]]] | None = None,
    levels: int = 2,
) -> list[dict]:
    cluster = cluster or cluster_chunks
    nodes = [dict(chunk) for chunk in chunks]
    current = nodes
    for level in range(1, levels):
        groups = cluster(current)
        summaries: list[dict] = []
        for index, members in enumerate(groups):
            if not members:
                continue
            text = "\n\n".join(current[i]["text"] for i in members)
            summaries.append(
                {
                    "text": summarize(text[:SUMMARY_CHAR_LIMIT]),
                    "source": "summary",
                    "page": 0,
                    "level": level,
                    "parent_id": f"L{level}-{index}",
                }
            )
        nodes.extend(summaries)
        current = summaries
        if len(current) < 2:
            break
    return nodes


def cluster_chunks(nodes: list[dict], max_clusters: int = 20) -> list[list[int]]:
    """shortcut: TF-IDF vectors instead of embedding vectors, so clustering needs no model calls; swap in embeddings when summary quality matters."""
    if len(nodes) < 5:
        return [list(range(len(nodes)))] if nodes else []

    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectors = TfidfVectorizer(stop_words="english").fit_transform(
        [n["text"] for n in nodes]
    ).toarray()

    count = max(2, min(max_clusters, len(nodes) // 10 or 2))
    if count == 1:
        return [list(range(len(nodes)))]

    reduced = _reduce(vectors)
    labels = KMeans(n_clusters=count, n_init=10, random_state=0).fit_predict(reduced)
    groups: dict[int, list[int]] = {}
    for index, label in enumerate(labels):
        groups.setdefault(int(label), []).append(index)
    return list(groups.values())


def _reduce(vectors):
    import umap

    components = min(10, vectors.shape[0] - 1, vectors.shape[1])
    if components < 2:
        return vectors
    return umap.UMAP(n_components=components, random_state=0).fit_transform(vectors)
