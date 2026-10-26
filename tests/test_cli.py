from studybuddy.cli import _answer_fn


class FakeLLM:
    def invoke(self, prompt):
        class Message:
            content = "answer for " + prompt
        return Message()


def test_answer_fn_calls_the_model():
    assert _answer_fn(FakeLLM())("hi") == "answer for hi"
