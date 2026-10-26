import base64
from pathlib import Path
from typing import Callable

import pymupdf

MIN_IMAGE_W = 200
MIN_IMAGE_H = 150
VISUAL_DRAWINGS = 15
MAX_PAGE_TEXT = 1200
CAPTION_LIMIT = 400
DPI = 120

CAPTION_PROMPT = (
    "This is a lecture slide. Describe only its diagrams, charts, images and "
    "the labels inside them. Transcribe axis labels, legends and figure text. "
    "Ignore the ordinary bullet text. Answer in at most three sentences."
)


def page_has_visual(page: pymupdf.Page) -> bool:
    for image in page.get_images(full=True):
        if image[2] >= MIN_IMAGE_W and image[3] >= MIN_IMAGE_H:
            return True
    return (
        len(page.get_drawings()) >= VISUAL_DRAWINGS
        and len(page.get_text().strip()) < MAX_PAGE_TEXT
    )


def default_captioner(llm) -> Callable[[bytes], str]:
    from langchain_core.messages import HumanMessage

    def caption(png: bytes) -> str:
        encoded = base64.b64encode(png).decode()
        message = HumanMessage(
            content=[
                {"type": "text", "text": CAPTION_PROMPT},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{encoded}"},
                },
            ]
        )
        text = llm.invoke([message]).content
        return text.strip()[:CAPTION_LIMIT]

    return caption


def caption_pages(
    path: Path,
    file_sha: str,
    cache_dir: Path,
    captioner: Callable[[bytes], str],
) -> dict[int, str]:
    out: dict[int, str] = {}
    cache = Path(cache_dir) / file_sha
    with pymupdf.open(path) as doc:
        for index, page in enumerate(doc, start=1):
            if not page_has_visual(page):
                continue
            cached = cache / f"{index}.txt"
            if cached.exists():
                out[index] = cached.read_text()
                continue
            png = page.get_pixmap(dpi=DPI).tobytes("png")
            text = captioner(png)[:CAPTION_LIMIT]
            cache.mkdir(parents=True, exist_ok=True)
            cached.write_text(text)
            out[index] = text
    return out
