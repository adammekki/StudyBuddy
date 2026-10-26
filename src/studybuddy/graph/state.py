from typing import TypedDict


class ChatState(TypedDict, total=False):
    question: str
    history: list[dict]
    standalone: str
    step_back: str
    chunks: list[dict]
    answer: str
    source: str
    notices: list[str]
    attempts: int
