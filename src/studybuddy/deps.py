from dataclasses import dataclass

from langchain_ollama import ChatOllama, OllamaEmbeddings

from studybuddy.config import Settings


@dataclass
class Deps:
    llm: object
    judge: object
    embed: object


def make_deps(settings: Settings) -> Deps:
    llm = ChatOllama(model=settings.gen_model, temperature=0)
    judge = ChatOllama(model=settings.judge_model, temperature=0)
    embed = OllamaEmbeddings(model=settings.embed_model)
    return Deps(llm=llm, judge=judge, embed=embed)
