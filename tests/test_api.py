import json

from fastapi.testclient import TestClient

from studybuddy.api import AppState, create_app
from studybuddy.graph.build import GraphDeps, build_graph


def fake_state(answer="X is Y", source="slides", sources=None, empty=False):
    def run(question, history):
        if empty:
            return {"error": "No slides indexed. Put PDFs in the RAG folder."}
        return {"answer": answer, "source": source, "sources": sources or []}

    return AppState(run=run, status=lambda: {"state": "idle", "decks": []}, reindex=lambda: None)


def graph_state(deps):
    graph = build_graph(deps)

    def run(question, history):
        out = graph.invoke(
            {"question": question, "history": history, "attempts": 0}
        )
        return {
            "answer": out["answer"],
            "source": out["source"],
            "sources": (
                [{"deck": "a.pdf", "page": 3}] if out["source"] == "slides" else []
            ),
        }

    return AppState(run=run, status=lambda: {"state": "idle"}, reindex=lambda: None)


def graph_deps(**over):
    d = dict(
        retrieve=lambda q, k: [
            {"text": "X is Y", "source": "a.pdf", "page": 3, "score": 0.9}
        ],
        answer_llm=lambda p: "X is Y [a.pdf p.3]",
        answer_stream=lambda p: iter(["X ", "is ", "Y [a.pdf p.3]"]),
        judge=lambda p: True,
        rewrite=lambda q, h: q,
        step_back=lambda q: q,
        web_search=lambda q: [{"title": "t", "url": "https://e.com", "text": "Z"}],
    )
    d.update(over)
    return GraphDeps(**d)


def chat_events(state):
    response = TestClient(create_app(state)).post(
        "/chat", json={"message": "what is X?", "history": [], "session_id": "s"}
    )
    return [json.loads(line) for line in response.text.strip().splitlines()]


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


def test_web_answer_has_no_slide_sources():
    state = fake_state(answer="From the web:\nZ", source="web", sources=[{"deck": "a.pdf", "page": 1}])
    events = [json.loads(l) for l in TestClient(create_app(state)).post(
        "/chat", json={"message": "q", "history": [], "session_id": "s"}
    ).text.splitlines()]
    assert [e["type"] for e in events] == ["answer", "done"]


def test_refused_answer_is_an_error_event():
    state = fake_state(answer="Your slides do not cover this", source="refused")
    events = [json.loads(l) for l in TestClient(create_app(state)).post(
        "/chat", json={"message": "q", "history": [], "session_id": "s"}
    ).text.splitlines()]
    assert events[0]["type"] == "error"


def test_tokens_stream_before_sources_and_done():
    events = chat_events(graph_state(graph_deps()))
    types = [event["type"] for event in events]
    assert types[:3] == ["token", "token", "token"]
    assert types[-3:] == ["answer", "sources", "done"]
    assert "".join(e["text"] for e in events if e["type"] == "token") == (
        "X is Y [a.pdf p.3]"
    )


def test_rejected_answer_is_reset_before_the_replacement():
    def answer_stream(prompt):
        return iter(["GOOD web"] if "WEB results" in prompt else ["BAD claim"])

    def judge(prompt):
        return "supported" not in prompt.lower()

    events = chat_events(
        graph_state(graph_deps(answer_stream=answer_stream, judge=judge))
    )
    types = [event["type"] for event in events]

    assert set(types[: types.index("reset")]) == {"token"}
    assert types.index("reset") < types.index("answer") < types.index("done")

    segments = [[]]
    for event in events:
        if event["type"] == "reset":
            segments.append([])
        elif event["type"] == "token":
            segments[-1].append(event["text"])
    assert "".join(segments[0]) == "BAD claim"
    assert "".join(segments[-1]) == "From the web:\nGOOD web"
    answer = next(event for event in events if event["type"] == "answer")
    assert answer["text"] == "From the web:\nGOOD web"


def test_deck_serves_pdf_inline_from_rag(tmp_path):
    (tmp_path / "deck one.pdf").write_bytes(b"%PDF-1.4 test")
    state = AppState(
        run=lambda q, h: {},
        status=lambda: {},
        reindex=lambda: None,
        rag_dir=tmp_path,
    )
    r = TestClient(create_app(state)).get("/deck/deck%20one.pdf")
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/pdf"
    assert r.headers["content-disposition"].startswith("inline")


def test_deck_refuses_paths_outside_rag(tmp_path):
    (tmp_path / "RAG").mkdir()
    state = AppState(
        run=lambda q, h: {},
        status=lambda: {},
        reindex=lambda: None,
        rag_dir=tmp_path / "RAG",
    )
    client = TestClient(create_app(state))
    assert client.get("/deck/..%2F..%2Fetc%2Fpasswd").status_code == 404
    assert client.get("/deck/missing.pdf").status_code == 404
