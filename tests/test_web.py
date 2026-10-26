from studybuddy import web


class FakeDDGS:
    def text(self, query, max_results=5):
        return [{"title": "t", "href": "https://x.test", "body": "snippet text"}]


def test_search_maps_snippets(monkeypatch):
    monkeypatch.setattr(web, "DDGS", FakeDDGS)
    assert web.web_search("q") == [
        {"title": "t", "url": "https://x.test", "text": "snippet text"}
    ]


def test_search_failure_returns_empty(monkeypatch):
    class Boom:
        def text(self, *a, **k):
            raise RuntimeError("blocked")

    monkeypatch.setattr(web, "DDGS", Boom)
    assert web.web_search("q") == []


def test_page_fetch_is_capped_and_tag_stripped(monkeypatch):
    class Response:
        text = "<p>" + "word " * 900 + "</p>"

        def raise_for_status(self):
            pass

    monkeypatch.setattr(web, "DDGS", FakeDDGS)
    monkeypatch.setattr(web.httpx, "get", lambda *a, **k: Response())
    out = web.web_search("q")
    assert len(out[0]["text"]) <= 4000
    assert "<p>" not in out[0]["text"]


def test_non_thin_snippets_skip_page_fetch(monkeypatch):
    class LongDDGS:
        def text(self, query, max_results=5):
            return [{"title": "t", "href": "https://x.test", "body": "y" * 500}]

    monkeypatch.setattr(web, "DDGS", LongDDGS)
    monkeypatch.setattr(web.httpx, "get", lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not fetch")))
    assert web.web_search("q")[0]["text"] == "y" * 500
