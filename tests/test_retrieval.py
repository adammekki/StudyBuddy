from studybuddy.retrieval import is_relevant, merge_results


def test_gate_requires_chunk_and_threshold():
    assert is_relevant([]) is False
    assert is_relevant([{"score": 0.2}]) is False
    assert is_relevant([{"score": 0.9}]) is True


def test_merge_dedupes_keeping_best_score():
    a = [{"text": "x", "source": "a.pdf", "page": 1, "level": 0, "score": 0.8}]
    b = [
        {"text": "x", "source": "a.pdf", "page": 1, "level": 0, "score": 0.6},
        {"text": "y", "source": "b.pdf", "page": 2, "level": 1, "score": 0.5},
    ]
    merged = merge_results(a, b)
    assert [m["text"] for m in merged] == ["x", "y"]
    assert merged[0]["score"] == 0.8
