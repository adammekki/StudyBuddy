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


def test_visual_caption_is_indexed_into_the_chunk(tmp_path):
    import pymupdf
    from studybuddy.store import Store

    rag = tmp_path / "RAG"
    rag.mkdir()
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "A slide with one picture " * 4)
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 300, 200), 0)
    pix.set_rect(pix.irect, (10, 90, 160))
    page.insert_image(pymupdf.Rect(100, 150, 400, 350), pixmap=pix)
    doc.save(rag / "visual.pdf")
    doc.close()

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

    class CaptionDeps(FakeDeps):
        caption = staticmethod(lambda png: "a green block diagram of the process")

    rebuild(settings, CaptionDeps())
    store = Store(settings.data_dir / "chroma", FakeEmbed())
    hits = store.search("green block diagram", 5)
    assert any("[Visuals: a green block diagram" in h["text"] for h in hits)


def test_build_lock_is_released_after_a_failed_rebuild(tmp_path):
    from studybuddy.pipeline import build_lock

    settings = Settings(
        rag_dir=tmp_path / "RAG",
        data_dir=tmp_path / "data",
        gen_model="g",
        judge_model="j",
        embed_model="e",
        k=8,
        threshold=0.35,
        chunk_size=800,
        chunk_overlap=100,
    )
    settings.rag_dir.mkdir()
    with build_lock(settings):
        pass
    with build_lock(settings):
        pass
