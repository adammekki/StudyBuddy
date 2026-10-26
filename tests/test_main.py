import pytest

from studybuddy import main


def test_model_check_reports_missing_model(monkeypatch):
    monkeypatch.setattr(main, "list_models", lambda: ["other:latest"])
    with pytest.raises(RuntimeError) as error:
        main.check_models(["qwen2.5:7b-instruct"])
    assert "ollama pull qwen2.5:7b-instruct" in str(error.value)


def test_untagged_model_name_matches_latest(monkeypatch):
    monkeypatch.setattr(main, "list_models", lambda: ["nomic-embed-text:latest"])
    main.check_models(["nomic-embed-text"])


def test_ollama_down_is_a_clear_error(monkeypatch):
    import httpx

    def boom(*a, **k):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(main.httpx, "get", boom)
    with pytest.raises(RuntimeError) as error:
        main.list_models()
    assert "ollama serve" in str(error.value)
