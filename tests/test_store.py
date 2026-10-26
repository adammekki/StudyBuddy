from studybuddy.store import Store


class FakeEmbed:
    def embed_documents(self, texts):
        return [[float(len(t) % 7), 1.0, 0.5] for t in texts]

    def embed_query(self, text):
        return [float(len(text) % 7), 1.0, 0.5]


def test_build_activate_and_old_index_survives_failed_build(tmp_path):
    s = Store(tmp_path / "chroma", FakeEmbed())
    v1 = s.build([{"text": "alpha", "source": "a.pdf", "page": 1, "level": 0}])
    s.activate(v1)
    assert s.active().name == v1
    assert s.search("alpha", 3)[0]["source"] == "a.pdf"


def test_search_score_is_similarity(tmp_path):
    s = Store(tmp_path / "chroma", FakeEmbed())
    s.activate(s.build([{"text": "beta", "source": "b.pdf", "page": 2, "level": 0}]))
    assert 0.0 <= s.search("beta", 1)[0]["score"] <= 1.0
