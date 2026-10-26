def is_relevant(chunks: list[dict], threshold: float = 0.35) -> bool:
    if not chunks:
        return False
    return max(chunk["score"] for chunk in chunks) >= threshold


def merge_results(a: list[dict], b: list[dict], limit: int = 8) -> list[dict]:
    best: dict[tuple, dict] = {}
    for chunk in [*a, *b]:
        key = (chunk["source"], chunk["page"], chunk["text"])
        current = best.get(key)
        if current is None or chunk["score"] > current["score"]:
            best[key] = chunk
    return sorted(best.values(), key=lambda c: c["score"], reverse=True)[:limit]


def format_context(chunks: list[dict], limit: int = 8) -> str:
    lines = []
    for chunk in chunks[:limit]:
        if chunk.get("level", 0) > 0:
            continue  # summaries carry no deck/page, so they cannot be cited
        lines.append(f"[{chunk['source']} p.{chunk['page']}] {chunk['text']}")
    return "\n\n".join(lines)
