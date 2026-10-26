from studybuddy.config import load_settings
from studybuddy.deps import make_deps


def test_models_are_gemma4_with_reasoning_off():
    deps = make_deps(load_settings())
    assert deps.llm.model == "gemma4:e4b"
    assert deps.llm.reasoning is False
    assert deps.judge.model == "gemma4:e4b"
    assert deps.judge.reasoning is False


def test_captioner_is_wired():
    assert callable(make_deps(load_settings()).caption)
