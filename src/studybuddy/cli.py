import argparse
import json
import sys

from studybuddy.config import load_settings
from studybuddy.deps import make_deps
from studybuddy.graph.build import GraphDeps, build_graph
from studybuddy.pipeline import index_status, rebuild
from studybuddy.store import Store
from studybuddy.web import web_search


def _text_of(message) -> str:
    return message.content if hasattr(message, "content") else str(message)


def _judge(llm):
    def judge(prompt: str) -> bool:
        answer = _text_of(llm.invoke(prompt)).strip().lower()
        if answer.startswith("yes"):
            return True
        if answer.startswith("no"):
            return False
        return True  # unparseable counts as pass, so the graph cannot loop

    return judge


def _answer_fn(llm):
    return lambda prompt: _text_of(llm.invoke(prompt))


def make_graph(settings, deps):
    store = Store(settings.data_dir / "chroma", deps.embed)
    answer = _answer_fn(deps.llm)
    return build_graph(
        GraphDeps(
            retrieve=store.search,
            answer_llm=answer,
            judge=_judge(deps.judge),
            rewrite=lambda q, h: _text_of(
                deps.judge.invoke(
                    "Rewrite the last question to stand alone. Output only the question.\n\n"
                    + "\n".join(f"{t['role']}: {t['content']}" for t in h[-6:])
                    + f"\n\nLast question: {q}"
                )
            ),
            step_back=lambda q: _text_of(
                deps.judge.invoke(
                    "Write the more general question behind this one. Output only the question.\n\n"
                    f"Question: {q}"
                )
            ),
            web_search=web_search,
            k=settings.k,
            threshold=settings.threshold,
            answer_stream=lambda prompt: (
                chunk.content for chunk in deps.llm.stream(prompt)
            ),
        )
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="studybuddy")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("index", help="index the RAG folder")
    ask = sub.add_parser("ask", help="ask one question")
    ask.add_argument("question")
    sub.add_parser("status", help="show index status")
    args = parser.parse_args(argv)

    settings = load_settings()

    if args.command == "status":
        print(json.dumps(index_status(settings), indent=2))
        return 0

    deps = make_deps(settings)

    if args.command == "index":
        print(json.dumps(rebuild(settings, deps), indent=2))
        return 0

    graph = make_graph(settings, deps)
    result = graph.invoke({"question": args.question, "history": [], "attempts": 0})
    print(result.get("answer", ""))
    print(f"\nsource: {result.get('source')}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
