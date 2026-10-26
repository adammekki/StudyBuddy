import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

STATIC_DIR = Path(__file__).parent / "static"


@dataclass
class AppState:
    run: Callable[[str, list[dict]], dict]
    status: Callable[[], dict]
    reindex: Callable[[], None]
    rag_dir: Path | None = None


class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []
    session_id: str = "default"


def _event(payload: dict) -> str:
    return json.dumps(payload) + "\n"


def create_app(state: AppState) -> FastAPI:
    app = FastAPI(title="MaiaGPT")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/status")
    def status() -> dict:
        return state.status()

    @app.get("/deck/{name}")
    def deck(name: str) -> FileResponse:
        root = (state.rag_dir or Path(".")).resolve()
        path = (root / name).resolve()
        if (
            not path.is_relative_to(root)
            or path.suffix.lower() != ".pdf"
            or not path.is_file()
        ):
            raise HTTPException(status_code=404, detail="deck not found")
        return FileResponse(
            path,
            media_type="application/pdf",
            filename=path.name,
            content_disposition_type="inline",
        )

    @app.post("/reindex", status_code=202)
    def reindex() -> dict:
        state.reindex()
        return {"state": "building"}

    @app.post("/chat")
    def chat(payload: ChatRequest) -> StreamingResponse:
        def stream():
            try:
                result = state.run(payload.message, payload.history)
            except Exception as error:  # surface, never a bare traceback
                yield _event({"type": "error", "message": str(error)})
                yield _event({"type": "done"})
                return

            if result.get("error"):
                yield _event({"type": "error", "message": result["error"]})
                yield _event({"type": "done"})
                return

            if result.get("source") == "refused":
                yield _event({"type": "error", "message": result.get("answer", "")})
                yield _event({"type": "done"})
                return

            yield _event(
                {
                    "type": "answer",
                    "text": result.get("answer", ""),
                    "source": result.get("source", "slides"),
                }
            )
            if result.get("source") == "slides" and result.get("sources"):
                yield _event({"type": "sources", "items": result["sources"]})
            for notice in result.get("notices", []):
                yield _event({"type": "notice", "text": notice})
            yield _event({"type": "done"})

        return StreamingResponse(stream(), media_type="application/x-ndjson")

    return app
