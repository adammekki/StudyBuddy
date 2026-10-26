import pytest

from studybuddy import main


def test_model_check_reports_missing_model(monkeypatch):
    monkeypatch.setattr(main, "list_models", lambda: ["other:latest"])
    with pytest.raises(RuntimeError) as error:
        main.check_models(["qwen2.5:7b-instruct"])
    assert "ollama pull qwen2.5:7b-instruct" in str(error.value)
