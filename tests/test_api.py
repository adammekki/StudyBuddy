import json

from fastapi.testclient import TestClient

from studybuddy.api import AppState, create_app


def fake_state(answer="X is Y", source="slides", sources=None, empty=False):
    def run(question, history):
        if empty:
            return {"error": "No slides indexed. Put PDFs in the RAG folder."}
        return {"answer": answer, "source": source, "sources": sources or []}

    return AppState(run=run, status=lambda: {"state": "idle", "decks": []}, reindex=lambda: None)


def test_chat_emits_answer_then_sources_then_done():
    app = create_app(
        fake_state(sources=[{"deck": "a.pdf", "page": 3}])
    )
    r = TestClient(app).post(
        "/chat", json={"message": "what is X?", "history": [], "session_id": "s1"}
    )
    events = [json.loads(line) for line in r.text.strip().splitlines()]
    assert [e["type"] for e in events] == ["answer", "sources", "done"]


def test_empty_index_returns_notice_without_model_call():
    events = [
        json.loads(line)
        for line in TestClient(create_app(fake_state(empty=True)))
        .post("/chat", json={"message": "q", "history": [], "session_id": "s"})
        .text.splitlines()
    ]
    assert events[0]["type"] == "error"
    assert "RAG" in events[0]["message"]


def test_status_and_reindex_contract():
    c = TestClient(create_app(fake_state()))
    assert c.get("/status").json()["state"] in {"idle", "building"}
    assert c.post("/reindex").status_code == 202
