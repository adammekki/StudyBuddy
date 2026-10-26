from studybuddy.raptor import build_tree


def test_tree_has_summary_levels_and_parent_links():
    chunks = [
        {"text": f"chunk {i}", "source": "a.pdf", "page": i, "level": 0}
        for i in range(1, 7)
    ]
    tree = build_tree(
        chunks,
        summarize=lambda t: "SUM:" + t[:10],
        cluster=lambda cs: [[0, 1, 2], [3, 4, 5]],
        levels=2,
    )
    levels = {n["level"] for n in tree}
    assert levels == {0, 1}
    assert all(n["parent_id"] for n in tree if n["level"] > 0)
    assert all(n["text"].startswith("SUM:") for n in tree if n["level"] > 0)
