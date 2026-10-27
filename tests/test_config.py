from studybuddy.config import load_settings


def test_defaults_and_env_override(monkeypatch, tmp_path):
    monkeypatch.delenv("STUDYBUDDY_RAG_DIR", raising=False)
    s = load_settings()
    assert s.rag_dir.name == "RAG"
    assert s.k == 8 and s.threshold == 0.35
    assert s.chunk_size == 800 and s.chunk_overlap == 100
    monkeypatch.setenv("STUDYBUDDY_RAG_DIR", str(tmp_path))
    assert load_settings().rag_dir == tmp_path


def test_env_file_is_loaded(monkeypatch, tmp_path):
    monkeypatch.delenv("STUDYBUDDY_RAG_DIR", raising=False)
    target = tmp_path / "from-dotenv"
    (tmp_path / ".env").write_text(f"STUDYBUDDY_RAG_DIR={target}\n")
    monkeypatch.chdir(tmp_path)
    assert load_settings().rag_dir == target
