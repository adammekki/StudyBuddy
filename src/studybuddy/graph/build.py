from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Callable, Iterator

from langgraph.graph import END, StateGraph

from studybuddy.graph.state import ChatState
from studybuddy.retrieval import format_context, is_relevant, merge_results

REWRITE_PROMPT = """Rewrite the user's last question so it stands alone.
Keep the meaning. Output only the rewritten question.

Conversation:
{history}

Last question: {question}"""

STEP_BACK_PROMPT = """Write the more general question behind this specific one.
Keep it short and answerable from course material. Output only the question.

Question: {question}"""

SCOPE_PROMPT = """Is this question about this course's lecture material? Reply with one word, YES or NO.

Question: {question}

Verdict:"""
# Cite every claim as [deck.pdf p.N] using the excerpt's own citation.
SLIDE_PROMPT = """Answer the question using only the lecture slide excerpts below.
If the excerpts do not support an answer, reply exactly: NOT IN THE SLIDES.

Background question: {step_back}

Excerpts:
{context}

Question: {question}"""

REGEN_PROMPT = """{base}

Your previous answer failed verification: {problem}
Answer again using only the excerpts. Cite every claim."""

HALLUCINATION_PROMPT = """Are all claims in the answer supported by the excerpts? Reply with one word, YES or NO.

Excerpts:
{context}

Answer:
{answer}

Verdict:"""

ADEQUACY_PROMPT = """Is the answer responsive to the question? Reply with one word, YES or NO.

Question: {question}

Answer:
{answer}

Verdict:"""

WEB_PROMPT = """Answer the question from these WEB results only.
Say the answer comes from the web. List the source URLs.
If the results do not answer it, say so.

WEB results:
{results}

Question: {question}"""

WEB_LABEL = "From the web"
REFUSAL = "Your slides do not cover this, and web search failed."


@dataclass
class GraphDeps:
    retrieve: Callable[[str, int], list[dict]]
    answer_llm: Callable[[str], str]
    judge: Callable[[str], bool]
    rewrite: Callable[[str, list[dict]], str]
    step_back: Callable[[str], str]
    web_search: Callable[[str], list[dict]]
    k: int = 8
    threshold: float = 0.35
    answer_stream: Callable[[str], Iterator[str]] | None = None


_EVENTS: ContextVar[Callable[[dict], None] | None] = ContextVar(
    "studybuddy_events", default=None
)


@contextmanager
def emit_to(sink: Callable[[dict], None]):
    """Send progress events from the graph nodes to ``sink``."""
    token = _EVENTS.set(sink)
    try:
        yield
    finally:
        _EVENTS.reset(token)


def _emit(event: dict) -> None:
    sink = _EVENTS.get()
    if sink is not None:
        sink(event)


def _answer_text(d: GraphDeps, prompt: str) -> str:
    chunks = d.answer_stream(prompt) if d.answer_stream else [d.answer_llm(prompt)]
    parts: list[str] = []
    for piece in chunks:
        if not piece:
            continue
        parts.append(piece)
        _emit({"type": "token", "text": piece})
    return "".join(parts).strip()


def _history_text(history: list[dict]) -> str:
    return "\n".join(f"{turn['role']}: {turn['content']}" for turn in history[-6:])


def build_graph(d: GraphDeps):
    graph = StateGraph(ChatState)

    def rewrite(state: ChatState) -> dict:
        history = state.get("history") or []
        if not history:
            return {"standalone": state["question"]}
        text = d.rewrite(state["question"], history)
        return {"standalone": text.strip() or state["question"]}

    def step_back(state: ChatState) -> dict:
        try:
            text = d.step_back(state["standalone"]).strip()
        except Exception:
            text = ""
        return {"step_back": text or state["standalone"]}

    def retrieve(state: ChatState) -> dict:
        primary = d.retrieve(state["standalone"], d.k)
        background = state.get("step_back", "")
        if background and background != state["standalone"]:
            primary = merge_results(primary, d.retrieve(background, d.k), d.k)
        return {"chunks": primary}

    def generate(state: ChatState) -> dict:
        context = format_context(state["chunks"], d.k)
        base = SLIDE_PROMPT.format(
            step_back=state.get("step_back", ""),
            context=context,
            question=state["question"],
        )
        if state.get("attempts"):
            base = REGEN_PROMPT.format(
                base=base, problem=state.get("notices", ["unsupported claims"])[-1]
            )
        return {"answer": _answer_text(d, base), "source": "slides"}

    def web(state: ChatState) -> dict:
        if state.get("answer"):
            _emit({"type": "reset"})  # the slide answer this one replaces
        results = d.web_search(state["standalone"])
        if not results:
            return {"answer": REFUSAL, "source": "refused"}
        text = "\n\n".join(
            f"{r['title']} ({r['url']})\n{r['text']}" for r in results
        )
        _emit({"type": "token", "text": f"{WEB_LABEL}:\n"})
        answer = _answer_text(
            d, WEB_PROMPT.format(results=text, question=state["question"])
        )
        return {"answer": f"{WEB_LABEL}:\n{answer}", "source": "web"}

    def after_retrieve(state: ChatState) -> str:
        return "generate" if is_relevant(state["chunks"], d.threshold) else "web"

    def after_generate(state: ChatState) -> str:
        context = format_context(state["chunks"], d.k)
        supported = d.judge(
            HALLUCINATION_PROMPT.format(context=context, answer=state["answer"])
        )
        if not supported:
            if state.get("attempts", 0) < 1:
                return "regenerate"
            return "web"
        adequate = d.judge(
            ADEQUACY_PROMPT.format(question=state["question"], answer=state["answer"])
        )
        return "ok" if adequate else "web"

    def regenerate(state: ChatState) -> dict:
        _emit({"type": "reset"})  # the rejected answer must not stay on screen
        return {"attempts": state.get("attempts", 0) + 1}

    graph.add_node("rewrite", rewrite)
    graph.add_node("step_back", step_back)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_node("regenerate", regenerate)
    graph.add_node("web", web)

    graph.set_entry_point("rewrite")
    graph.add_edge("rewrite", "step_back")
    graph.add_conditional_edges(
        "step_back",
        lambda s: "retrieve" if d.judge(SCOPE_PROMPT.format(question=s["standalone"])) else "web",
        {"retrieve": "retrieve", "web": "web"},
    )
    graph.add_conditional_edges(
        "retrieve", after_retrieve, {"generate": "generate", "web": "web"}
    )
    graph.add_conditional_edges(
        "generate",
        after_generate,
        {"ok": END, "regenerate": "regenerate", "web": "web"},
    )
    graph.add_edge("regenerate", "generate")
    graph.add_edge("web", END)

    return graph.compile()
