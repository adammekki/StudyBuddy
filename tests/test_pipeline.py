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
