from studybuddy.config import Settings
from studybuddy.pipeline import index_status, rebuild


class _NullDeps:
    llm = None
    judge = None
    embed = None


def test_rebuild_reports_bad_pdf_and_keeps_old_collection(tmp_path):
    rag = tmp_path / "RAG"
    rag.mkdir()
    (rag / "good.pdf").write_bytes(b"%PDF-broken")
    settings = Settings(
        rag_dir=rag,
        data_dir=tmp_path / "data",
        gen_model="g",
        judge_model="j",
        embed_model="e",
        k=8,
        threshold=0.35,
        chunk_size=800,
        chunk_overlap=100,
    )
    result = rebuild(settings, deps=_NullDeps(), names={"good.pdf"})
    assert result["errors"]
    assert index_status(settings)["state"] in {"idle", "failed"}


class FakeLLM:
    def invoke(self, prompt):
        from types import SimpleNamespace

        return SimpleNamespace(content="summary of the lecture")


class FakeEmbed:
    def embed_documents(self, texts):
        return [[1.0, 2.0, 3.0] for _ in texts]

    def embed_query(self, text):
        return [1.0, 2.0, 3.0]


class FakeDeps:
    llm = FakeLLM()
    judge = None
    embed = FakeEmbed()


def make_pdf(path, text):
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    doc.save(path)
    doc.close()


def test_adding_one_deck_keeps_the_others(tmp_path):
    rag = tmp_path / "RAG"
    rag.mkdir()
    make_pdf(rag / "a.pdf", "alpha lecture material " * 5)
    make_pdf(rag / "b.pdf", "beta lecture material " * 5)
    settings = Settings(
        rag_dir=rag,
        data_dir=tmp_path / "data",
        gen_model="g",
        judge_model="j",
        embed_model="e",
        k=8,
        threshold=0.35,
        chunk_size=800,
        chunk_overlap=100,
    )
    assert rebuild(settings, FakeDeps())["decks"] == 2
    make_pdf(rag / "c.pdf", "gamma lecture material " * 5)
    second = rebuild(settings, FakeDeps(), names={"c.pdf"})
    assert second["decks"] == 3
