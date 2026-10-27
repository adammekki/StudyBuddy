import pytest

from studybuddy.graph.build import GraphDeps, build_graph, emit_to


def _deps(**over):
    d = dict(
        retrieve=lambda q, k: [
            {"text": "X is Y", "source": "a.pdf", "page": 3, "score": 0.9}
        ],
        answer_llm=lambda p: "X is Y [a.pdf p.3]",
        judge=lambda p: True,
        rewrite=lambda q, h: q,
        step_back=lambda q: q,
        web_search=lambda q: [{"title": "t", "url": "u", "text": "web says Z"}],
        k=8,
        threshold=0.35,
    )
    d.update(over)
    return GraphDeps(**d)


def test_relevant_question_answers_from_slides():
    out = build_graph(_deps()).invoke(
        {"question": "what is X?", "history": [], "attempts": 0}
    )
    assert out["source"] == "slides" and "a.pdf p.3" in out["answer"]


def test_weak_retrieval_goes_to_web_with_label():
    d = _deps(
        retrieve=lambda q, k: [
            {"text": "x", "source": "a.pdf", "page": 1, "score": 0.1}
        ],
        answer_llm=lambda p: ("web says Z" if "WEB" in p else "slide answer"),
    )
    out = build_graph(d).invoke({"question": "q", "history": [], "attempts": 0})
    assert out["source"] == "web"


def test_failed_hallucination_check_regenerates_once_then_web():
    def judge(p):
        return False if "supported" in p.lower() else True

    out = build_graph(
        _deps(judge=judge, answer_llm=lambda p: "unsupported claim")
    ).invoke({"question": "q", "history": [], "attempts": 0})
    assert out["source"] == "web"


def test_follow_up_is_rewritten_before_retrieval():
    seen = {}
    def retrieve(q, k):
        seen["q"] = q
        return [{"text": "X is Y", "source": "a.pdf", "page": 3, "score": 0.9}]

    build_graph(
        _deps(retrieve=retrieve, rewrite=lambda q, h: "what is X and why does it hold?")
    ).invoke(
        {"question": "why?", "history": [{"role": "user", "content": "what is X?"}], "attempts": 0}
    )
    assert seen["q"] == "what is X and why does it hold?"


def test_step_back_retrieves_both_queries_and_merges():
    seen = []
    def retrieve(q, k):
        seen.append(q)
        return [{"text": "ctx for " + q, "source": "a.pdf", "page": 1, "score": 0.9}]

    out = build_graph(
        _deps(retrieve=retrieve, step_back=lambda q: "general: " + q)
    ).invoke({"question": "q", "history": [], "attempts": 0})
    assert seen == ["q", "general: q"]
    assert out["source"] == "slides"


def test_step_back_failure_does_not_block_answer():
    def boom(q):
        raise RuntimeError("timeout")

    out = build_graph(_deps(step_back=boom)).invoke(
        {"question": "q", "history": [], "attempts": 0}
    )
    assert out["source"] == "slides"


def test_both_paths_fail_gives_labelled_refusal():
    d = _deps(retrieve=lambda q, k: [], web_search=lambda q: [])
    out = build_graph(d).invoke({"question": "q", "history": [], "attempts": 0})
    assert out["source"] == "refused"
    assert out["answer"]


def test_web_answer_is_labelled_and_carries_no_slide_text():
    d = _deps(
        retrieve=lambda q, k: [{"text": "SECRET SLIDE", "source": "a.pdf", "page": 1, "score": 0.1}],
        answer_llm=lambda p: ("web answer" if "WEB" in p else "slide answer"),
    )
    out = build_graph(d).invoke({"question": "q", "history": [], "attempts": 0})
    assert out["source"] == "web"
    assert out["answer"].startswith("From the web")
    assert "SECRET SLIDE" not in out["answer"]


def test_score_just_below_threshold_goes_to_web():
    d = _deps(retrieve=lambda q, k: [{"text": "x", "source": "a.pdf", "page": 1, "score": 0.349}])
    assert build_graph(d).invoke({"question": "q", "history": [], "attempts": 0})["source"] == "web"


def test_answer_llm_only_emits_one_token_event():
    events = []
    with emit_to(events.append):
        out = build_graph(_deps()).invoke(
            {"question": "q", "history": [], "attempts": 0}
        )
    assert events == [{"type": "token", "text": "X is Y [a.pdf p.3]"}]
    assert out["answer"] == "X is Y [a.pdf p.3]"
    assert out["source"] == "slides"


def test_regeneration_happens_exactly_once():
    calls = {"slide": 0, "web": 0}

    def answer(p):
        if "WEB results" in p:
            calls["web"] += 1
        else:
            calls["slide"] += 1
        return "unsupported claim"

    def judge(p):
        return False if "supported" in p.lower() else True

    build_graph(_deps(judge=judge, answer_llm=answer)).invoke(
        {"question": "q", "history": [], "attempts": 0}
    )
    assert calls["slide"] == 2
    assert calls["web"] == 1


def test_each_node_logs_a_line_to_stdout(capsys):
    build_graph(_deps()).invoke({"question": "what is X?", "history": [], "attempts": 0})
    out = capsys.readouterr().out
    for name in ("rewrite", "step_back", "route", "retrieve", "generate"):
        assert f"[{name}]" in out
