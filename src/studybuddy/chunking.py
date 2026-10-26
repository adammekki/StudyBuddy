from pathlib import Path

import pymupdf4llm
from langchain_text_splitters import RecursiveCharacterTextSplitter

MIN_CHARS = 40


class NoTextError(Exception):
    """The PDF produced no extractable text (for example, scanned slides)."""


def extract_pages(path: Path) -> list[tuple[int, str]]:
    pages = pymupdf4llm.to_markdown(str(path), page_chunks=True)
    result = [(i, chunk.get("text", "")) for i, chunk in enumerate(pages, start=1)]
    if not any(text.strip() for _, text in result):
        raise NoTextError(f"{path.name} has no extractable text")
    return result


def make_chunks(
    pages: list[tuple[int, str]],
    source: str,
    size: int = 800,
    overlap: int = 100,
) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size, chunk_overlap=overlap
    )
    chunks: list[dict] = []
    for page_number, text in pages:
        for index, part in enumerate(splitter.split_text(text)):
            if len(part) < MIN_CHARS:
                continue
            chunks.append(
                {
                    "text": part,
                    "source": source,
                    "page": page_number,
                    "chunk_index": index,
                    "level": 0,
                }
            )
    return chunks
