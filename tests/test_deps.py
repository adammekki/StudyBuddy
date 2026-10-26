from studybuddy.config import load_settings
from studybuddy.deps import make_deps


def test_generator_is_gemma4_with_reasoning_off():
    deps = make_deps(load_settings())
    assert deps.llm.model == "gemma4:e4b"
    assert deps.llm.reasoning is False
    assert deps.judge.model == "llama3.2:3b"
