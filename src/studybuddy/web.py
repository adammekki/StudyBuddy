import re

import httpx
from ddgs import DDGS

MAX_CHARS = 4000
TIMEOUT = 5.0
THIN_SNIPPETS = 400


def _fetch_text(url: str) -> str:
    response = httpx.get(url, timeout=TIMEOUT, follow_redirects=True)
    response.raise_for_status()
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", response.text, flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def web_search(query: str, fetch_pages: int = 2) -> list[dict]:
    try:
        raw = list(DDGS().text(query, max_results=5))
    except Exception:
        return []

    results = [
        {
            "title": r.get("title", ""),
            "url": r.get("href") or r.get("url", ""),
            "text": r.get("body", ""),
        }
        for r in raw
    ]

    if results and sum(len(r["text"]) for r in results) < THIN_SNIPPETS:
        for result in results[:fetch_pages]:
            try:
                fetched = _fetch_text(result["url"])[:MAX_CHARS]
            except Exception:
                continue
            if fetched:
                result["text"] = fetched

    return [r for r in results if r["text"]]
