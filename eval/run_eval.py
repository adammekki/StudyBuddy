import json
import re
import sys
from pathlib import Path

from studybuddy.cli import make_graph
from studybuddy.config import load_settings
from studybuddy.deps import make_deps
from studybuddy.pipeline import index_status

QUESTIONS = Path(__file__).parent / "questions.json"


def citations(answer: str) -> list[str]:
    return sorted(set(re.findall(r"\[([^\]]+)\]", answer)))


def main() -> int:
    settings = load_settings()
    status = index_status(settings)
    if not status["collection"]:
        print(f"No index found. Add PDFs to {settings.rag_dir} and run: studybuddy index")
        return 1

    questions = json.loads(QUESTIONS.read_text())["questions"]
    graph = make_graph(settings, make_deps(settings))

    for item in questions:
        result = graph.invoke(
            {"question": item["question"], "history": [], "attempts": 0}
        )
        print(f"\n=== {item['id']} ({item['kind']}) ===")
        print("Q:", item["question"])
        print("Expected:", item["expect"])
        print("Source:", result.get("source"))
        print("Answer:", result.get("answer", "").strip())
        print("Cited:", ", ".join(citations(result.get("answer", ""))) or "none")
    print("\nMark each: answer correct? citation correct? behaviour correct?")
    return 0


if __name__ == "__main__":
    sys.exit(main())
